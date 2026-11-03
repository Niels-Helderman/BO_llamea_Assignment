"""Problem factories: use IOH/Practical_Problems; never reimplement benchmarks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    """Serializable problem descriptor (safe to pass to spawned workers)."""
    kind: str
    dimension: int = 6
    fid: int = 1
    instance: int = 1
    problem_id: int = 1124


def from_config(cfg: dict, kind: str) -> ProblemSpec:
    """Input: config and 'bbob'/'realworld'. Output: validated problem spec."""
    d = int(cfg['dimension'])
    if d != 6:
        raise ValueError('Assignment requires dimension=6')
    if kind == 'bbob':
        p = cfg['problems']['bbob']
        fid, instance = int(p['fid']), int(p['instance'])
        if fid not in range(1, 25) or instance != 1:
            raise ValueError('BBOB requires fid 1..24 and instance=1')
        return ProblemSpec(kind, d, fid=fid, instance=instance)
    if kind == 'realworld':
        pid = int(cfg['problems']['realworld']['problem_id'])
        if pid not in (1124, 1125, 1129, 1130):
            raise ValueError('This real-world problem ID does not support dimension 6')
        return ProblemSpec(kind, d, problem_id=pid)
    if kind == 'sphere':  # offline dependency-free smoke testing only
        return ProblemSpec(kind, d)
    raise ValueError(f'Unknown problem kind: {kind}')


class _Sphere:
    """Six-dimensional toy objective; never used for actual assignment results."""
    def __init__(self, dim: int):
        self.bounds = type('Bounds', (), {'lb': np.full(dim, -5.), 'ub': np.full(dim, 5.)})()
        self.optimum = type('Optimum', (), {'y': 0.})()

    def __call__(self, x):
        return float(np.dot(x, x))


def make_problem(spec: ProblemSpec):
    """Input: ProblemSpec. Output: fresh callable objective with IOH-style bounds."""
    if spec.kind == 'sphere':
        return _Sphere(spec.dimension)
    if spec.kind == 'bbob':
        import ioh
        return ioh.get_problem(spec.fid, instance=spec.instance,
                               dimension=spec.dimension, problem_class=ioh.ProblemClass.BBOB)
    if spec.kind == 'realworld':
        from Practical_Problems import get_practical_problem
        return get_practical_problem(spec.problem_id, dim=spec.dimension)
    raise ValueError(spec.kind)


def bounds_of(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Input: IOH-style problem. Output: validated lower/upper bound vectors."""
    lower = np.broadcast_to(np.asarray(problem.bounds.lb, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(problem.bounds.ub, dtype=float), (dim,)).copy()
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError('Invalid problem bounds')
    return lower, upper


def known_optimum(spec: ProblemSpec) -> float | None:
    """Return a justified fixed AOCC reference, or None if unavailable.

    IOH's BBOB problems expose their actual instance-specific optima. However,
    Practical_Problems hardcodes ``optimum.y = 0`` as a PLACEHOLDER for every
    real-world problem. Do not treat that placeholder as a known optimum.

    Exception: problem 1124 (Frequency Modulated Sound Waves) explicitly sums
    squared residuals, so 0 is a proven lower bound and usable AOCC reference.
    Other real-world problems require a documented reference in the YAML.
    """
    if spec.kind == 'realworld':
        return 0.0 if spec.problem_id == 1124 else None
    value = getattr(getattr(make_problem(spec), 'optimum', None), 'y', None)
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def metric_for_target(cfg: dict, kind: str, cache_dir) -> dict:
    """Fixed, independently calibrated per-problem AOCC metric.

    BBOB supplies an instance-specific true optimum; problem 1124 supplies a
    proven lower bound of zero. Objective *scale* is not provided by either
    wrapper. Estimate it once from a separately logged, fixed-seed space-filling
    design, not from any candidate algorithm's trajectory. Calibration calls
    are setup overhead, outside each independently counted optimizer run.
    """
    from dataclasses import asdict
    import hashlib
    import json
    from pathlib import Path
    from .utils import dump_json

    spec = from_config(cfg, kind)
    settings = cfg['stage1']['aocc']
    ref = settings[f'{kind}_reference']
    if ref == 'auto':
        ref = known_optimum(spec)
    if ref is None:
        raise ValueError(
            f'No justified AOCC reference for {kind} problem ID {spec.problem_id}. '
            'Provide a fixed, documented stage1.aocc.realworld_reference; '
            'Practical_Problems.optimum.y=0 is generally only a placeholder.'
        )
    reference = float(ref)
    lower = float(settings['lower_log10_error'])
    upper = float(settings['upper_log10_error'])
    points = int(settings.get('calibration_points', 32))
    seed = int(settings.get('calibration_seed', 2026))
    scale_mode = settings.get('scale', 'auto')
    if not np.isfinite(reference) or not np.isfinite(lower + upper) or lower >= upper:
        raise ValueError('Invalid AOCC reference or log-error window')
    if not (4 <= points <= 1024):
        raise ValueError('calibration_points must be 4..1024')

    identity = dict(problem=asdict(spec), reference=reference, lower=lower,
                    upper=upper, points=points, seed=seed, scale=scale_mode)
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'{kind}.json'
    if path.exists():
        stored = json.loads(path.read_text(encoding='utf-8'))
        if stored.get('signature') == signature:
            scale = float(stored['metric']['scale'])
            if scale > 0 and np.isfinite(scale):
                return stored['metric']

    observed = []
    if scale_mode == 'auto':
        problem = make_problem(spec)
        lb, ub = bounds_of(problem, spec.dimension)
        rng = np.random.default_rng(seed)
        # Independent deterministic calibration design. NEVER re-use evaluation
        # seeds or regenerate this scale per candidate.
        for x in rng.uniform(lb, ub, size=(points, spec.dimension)):
            observed.append(float(problem(x)))
        if not np.isfinite(observed).all():
            raise ValueError('Calibration produced nonfinite objective values')
        if min(observed) < reference - 1e-8:
            raise ValueError('AOCC reference is above a calibrated observation')
        positive = np.asarray(observed) - reference
        scale = float(np.median(positive))
        if scale <= 1e-12:
            scale = float(np.max(positive))
        if scale <= 1e-12:
            raise ValueError('Cannot infer positive objective scale; specify stage1.aocc.scale')
    else:
        scale = float(scale_mode)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Explicit AOCC scale must be finite and positive')

    metric = dict(reference=reference, scale=scale, lower=lower, upper=upper)
    dump_json(path, dict(signature=signature, settings=identity, metric=metric,
                         calibration_observations=observed,
                         note='Setup-only objective queries; excluded from every BO budget.'))
    return metric
