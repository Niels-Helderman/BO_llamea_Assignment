"""Single-run execution, strict objective accounting, AOCC, and seed parallelism.

Every algorithm (generated or baseline) uses the same interface:
    Algorithm(budget, dim, **options)(objective) -> (best_y, best_x)
The returned best is informative; all reported performance uses RECORDED f(x).

Generated code runs in a subprocess with timeout. A subprocess is NOT a
security boundary: run the full project inside a restricted VM/container.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import os
import time
import traceback
import warnings
from collections import Counter
from typing import Any
import numpy as np

from .problems import ProblemSpec, make_problem, bounds_of


@dataclass(frozen=True)
class Algorithm:
    """Portable optimizer descriptor. For generated code provide name AND source."""
    name: str
    code: str | None = None
    class_name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Output of one seeded run; history lists actual completed objective calls."""
    algorithm: str
    problem: str
    seed: int
    budget: int
    xs: list[list[float]] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    status: str = 'error'
    error: str = ''
    seconds: float = 0.0
    aocc: float | None = None
    warning_count: int = 0
    warnings: list[dict] = field(default_factory=list)

    @property
    def best(self) -> float | None:
        return min(self.ys) if self.ys else None

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetExceeded(RuntimeError):
    pass


def best_so_far(ys, budget: int | None = None) -> np.ndarray:
    """Input: minimization observations and optional budget. Output: best-so-far curve.

    With budget, unspent evaluations repeat the last best objective (for AOCC
    horizon comparability); no fake objective observations are logged.
    """
    ys = np.asarray(ys, dtype=float).reshape(-1)
    if not len(ys) or not np.isfinite(ys).all():
        raise ValueError('Nonempty, finite objective history required')
    curve = np.minimum.accumulate(ys)
    if budget is not None:
        if len(ys) > budget or budget <= 0:
            raise ValueError('Invalid budget or history exceeds budget')
        curve = np.pad(curve, (0, budget - len(curve)), constant_values=curve[-1])
    return curve


def aocc(ys, budget: int, reference: float, lower=-8.0, upper=2.0,
         scale: float | None = 1.0) -> float:
    """Input: history, horizon, fixed optimum/reference and log-error bounds.

    Output: normalized area-over-convergence (larger is better, range [0,1]).
    Uses mean of clipped normalized negative log10((best_y-reference)/scale).
    The fixed reference, optional scale and precision window determine the
    score transformation. ``scale=None`` applies unscaled log-error AOCC.
    """
    if (not np.isfinite(reference) or not np.isfinite(lower + upper) or lower >= upper):
        raise ValueError('Invalid AOCC reference/range')
    curve = best_so_far(ys, budget)
    if scale is None:
        # Transform unscaled best-so-far error with the fixed log window.
        precision = np.maximum(curve - reference, 10.0 ** lower)
        clipped = np.clip(np.log10(precision), lower, upper)
        return float((1.0 - (clipped - lower) / (upper - lower)).mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid AOCC scale')
    error = np.maximum((curve - reference) / scale, 10.0 ** lower)
    return float(np.clip((upper - np.log10(error)) / (upper - lower), 0, 1).mean())


def _optimizer_class(spec: Algorithm):
    """Input: generated Python source/class name OR a named baseline.

    Output: a Python optimizer class. Generated code arrives from LLaMEA as
    a STRING, not a module. Compile+exec defines its class inside this worker;
    the class is retrieved by the name returned by the LLM. This is
    untrusted code execution: the subprocess timeout is NOT a sandbox.
    """
    if spec.code is None:
        from .baselines import get_baseline
        return get_baseline(spec.name)
    if not spec.class_name:
        raise ValueError('Missing generated class name')
    namespace = {'__name__': 'llamea_candidate'}
    # Running exec defines the class; it also executes any top-level statements.
    exec(compile(spec.code, '<generated candidate>', 'exec'), namespace)
    cls = namespace.get(spec.class_name)
    if not isinstance(cls, type):
        raise ValueError(f'Generated class {spec.class_name!r} not found')
    return cls


def _seed(seed: int, torch_needed: bool):
    """Seed global RNGs; avoid importing slow PyTorch for NumPy-only candidates."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch_needed:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(1)


def _execute(spec: Algorithm, problem_spec: ProblemSpec, seed: int, budget: int,
             aocc_config: dict | None, ioh_root: str | None) -> Result:
    """Child-side run. Input: optimizer/problem/seed/budget; output: Result."""
    result = Result(spec.name, problem_spec.kind, seed, budget)
    start = time.monotonic()
    ioh_logger = None
    problem = None
    try:
        # Limit numerical-library threads in each evaluation worker.
        for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        src = (spec.code or '').lower()
        _seed(seed, spec.name == 'botorch' or 'torch' in src or 'botorch' in src)
        problem = make_problem(problem_spec)
        lb, ub = bounds_of(problem, problem_spec.dimension)
        if ioh_root is not None and problem_spec.kind != 'sphere':
            import ioh
            # Optional native IOH output supplements per-call project traces.
            ioh_logger = ioh.logger.Analyzer(
                triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
                root=ioh_root, folder_name=f'{problem_spec.kind}_{spec.name}_seed{seed}',
                algorithm_name=spec.name, store_positions=False)
            problem.attach_logger(ioh_logger)

        bounds = type('Bounds', (), {'lb': lb, 'ub': ub})()
        violated = False
        # Guard and record every objective evaluation.
        def objective(x):
            nonlocal violated
            # Count exactly one accepted vector query per objective call.
            x = np.asarray(x, dtype=float)
            if x.shape != (problem_spec.dimension,) or not np.isfinite(x).all():
                raise ValueError(f'Query must be finite with shape ({problem_spec.dimension},)')
            if np.any(x < lb) or np.any(x > ub):
                raise ValueError('Out-of-bounds objective query')
            if len(result.ys) >= budget:
                violated = True
                raise BudgetExceeded(f'Attempted more than {budget} objective calls')
            y = float(problem(x))
            if not np.isfinite(y):
                raise ValueError('Objective returned nonfinite value')
            result.xs.append(x.tolist())
            result.ys.append(y)
            return y

        objective.bounds = bounds
        objective.dim = problem_spec.dimension
        objective.dimension = problem_spec.dimension
        # Convert source text to class -> make a fresh optimizer -> run BO.
        # All random initialization and BO-chosen points call objective(x).
        # Aggregate repeated numerical warnings inside each worker.
        warning_counts = Counter()
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            def capture_warning(message, category, filename, lineno, file=None, line=None):
                key = (category.__name__, str(message)[:300])
                warning_counts[key] += 1
            warnings.showwarning = capture_warning
            try:
                optimizer = _optimizer_class(spec)(budget=budget, dim=problem_spec.dimension, **spec.options)
                optimizer(objective)
            finally:
                result.warning_count = sum(warning_counts.values())
                result.warnings = [dict(category=category, message=message, count=count)
                                   for (category, message), count in warning_counts.most_common(8)]
        if violated:
            result.status, result.error = 'invalid', 'Budget exceeded, even if exception was caught'
        elif not result.ys:
            result.status, result.error = 'invalid', 'No objective calls made'
        else:
            result.status = 'ok'
            if aocc_config:
                result.aocc = aocc(result.ys, budget, float(aocc_config['reference']),
                                   float(aocc_config['lower']), float(aocc_config['upper']),
                                   (float(aocc_config['scale'])
                                    if aocc_config.get('scale') is not None else None))
    except BaseException as exc:
        result.status = 'invalid' if isinstance(exc, BudgetExceeded) else 'error'
        result.error = traceback.format_exc(limit=5)[-1800:]
    finally:
        result.seconds = time.monotonic() - start
        if ioh_logger is not None:
            try:
                problem.detach_logger()
                ioh_logger.close()
            except Exception as exc:
                if result.status == 'ok':
                    result.status, result.error = 'error', f'IOH logger cleanup: {exc}'
    return result


def _worker(conn, spec, problem, seed, budget, metric, ioh_root):
    """Process entry; sends a serializable result through a one-way Pipe."""
    try:
        conn.send(_execute(spec, problem, seed, budget, metric, ioh_root).to_dict())
    except BaseException:
        conn.send(Result(spec.name, problem.kind, seed, budget,
                         error=traceback.format_exc(limit=5)[-1800:]).to_dict())
    finally:
        conn.close()


def evaluate_many(spec: Algorithm, problem: ProblemSpec, seeds: list[int], budget: int,
                  timeout: float, workers: int = 1, aocc_config: dict | None = None,
                  ioh_root: Path | None = None) -> list[Result]:
    """Input: optimizer/problem/seeds/budget; output: results in seed order.

    Spawn at most `workers` independent processes. Each has its own hard wall
    timeout, fresh objective state, and RNGs. Avoid nested LLaMEA parallelism.
    """
    if not seeds or len(set(seeds)) != len(seeds) or budget < 1 or timeout <= 0 or workers < 1:
        raise ValueError('Invalid seeds, budget, timeout or worker count')
    if ioh_root is not None:
        Path(ioh_root).mkdir(parents=True, exist_ok=True)
    # Restrict BLAS threads before launching spawned workers.
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    ctx = mp.get_context('spawn')
    pending = list(seeds)
    active: dict[int, tuple] = {}
    results: dict[int, Result] = {}
    workers = min(workers, len(seeds))

    def stop(proc):
        proc.join(timeout=0.1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)

    try:
        while pending or active:
            while pending and len(active) < workers:
                seed = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_worker, args=(send, spec, problem, seed, budget,
                    aocc_config, str(ioh_root) if ioh_root is not None else None))
                started = time.monotonic()
                try:
                    proc.start()
                    send.close()
                    active[seed] = (proc, recv, started)
                except BaseException as exc:
                    recv.close()
                    send.close()
                    results[seed] = Result(spec.name, problem.kind, seed, budget,
                                           error=f'Process startup failed: {exc}')
            if not active:
                continue
            ready = wait([entry[1] for entry in active.values()], timeout=0.1)
            for seed, (proc, conn, started) in list(active.items()):
                if conn in ready:
                    try:
                        results[seed] = Result(**conn.recv())
                    except (EOFError, OSError, TypeError, ValueError) as exc:
                        results[seed] = Result(spec.name, problem.kind, seed, budget,
                                               error=f'Worker exited without valid result: {exc}')
                    conn.close()
                    stop(proc)
                    del active[seed]
                elif time.monotonic() - started > timeout:
                    stop(proc)
                    conn.close()
                    results[seed] = Result(spec.name, problem.kind, seed, budget,
                        status='timeout', error=f'Exceeded {timeout}s wall time', seconds=timeout)
                    del active[seed]
    finally:
        for proc, conn, _ in active.values():
            stop(proc)
            conn.close()
    return [results[s] for s in seeds]


def evaluate_one(spec: Algorithm, problem: ProblemSpec, seed: int, budget: int,
                 timeout: float = 180, aocc_config: dict | None = None) -> Result:
    """Input: one seed; output: one result. Uses the same isolated worker path."""
    return evaluate_many(spec, problem, [seed], budget, timeout, 1, aocc_config)[0]


def rows(result: Result, optimum: float | None = None) -> list[dict]:
    """Input: result; output: long-format rows for real objective evaluations."""
    best = float('inf')
    output = []
    for i, (x, y) in enumerate(zip(result.xs, result.ys), start=1):
        best = min(best, y)
        row = dict(method=result.algorithm, problem=result.problem, seed=result.seed,
                   evaluation=i, objective=y, best_so_far=best, status=result.status)
        if optimum is not None:
            row['loss'] = best - optimum
        row.update({f'x{j + 1}': value for j, value in enumerate(x)})
        output.append(row)
    return output
