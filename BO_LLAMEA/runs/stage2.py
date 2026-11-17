"""Benchmark both Stage 1 winners, CMA-ES and BoTorch on both problems.

Quick preview: python -m runs.stage2 --budget 50
Final run:     python -m runs.stage2 --budget 600
Each method/problem pairing uses the same ten group-specific seeds.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

from src.problems import from_config, known_optimum, metric_for_target
from src.performance import Algorithm, evaluate_many, rows
from src.utils import (DEFAULT_CONFIG, RESULTS, dump_json, group_seeds,
                       load_config, metadata, plot_stage2, save_csv)


def load_winner(target: str) -> Algorithm:
    """Load a selected generated algorithm, without modifying its source code."""
    folder = RESULTS / 'stage1' / target
    code = (folder / 'best.py').read_text(encoding='utf-8')
    info = json.loads((folder / 'best.json').read_text(encoding='utf-8'))
    return Algorithm(f'LLaMEA_{target}', code=code, class_name=info['class_name'])


def summarize(runs, problem: str) -> dict:
    """All 10 seeded results, with failure counts and valid-run uncertainty."""
    good = [r for r in runs if r.status == 'ok']
    final = [r.best for r in good]
    scores = [r.aocc for r in good if r.aocc is not None]
    return dict(problem=problem, algorithm=runs[0].algorithm,
                runs=len(runs), valid=len(good), failed=len(runs) - len(good),
                final_mean=float(np.mean(final)) if final else None,
                final_se=float(np.std(final, ddof=1)/np.sqrt(len(final))) if len(final) > 1 else None,
                mean_aocc=float(np.mean(scores)) if scores else None,
                warning_count=sum(r.warning_count for r in runs),
                objective_calls=sum(len(r.ys) for r in runs),
                total_worker_seconds=round(sum(r.seconds for r in runs), 3))


def main():
    """Save each benchmark under its configured or overridden evaluation budget."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--budget', type=int, default=None,
                   help='Exact per-run objective call maximum (1..600); defaults to YAML')
    p.add_argument('--workers', type=int, default=None,
                   help='Independent seeded runs in parallel; defaults to stage2.seed_workers')
    args = p.parse_args()
    cfg = load_config(args.config)
    stage = cfg['stage2']
    seeds = group_seeds(cfg)
    budget = int(stage['budget'] if args.budget is None else args.budget)
    workers = int(stage['seed_workers'] if args.workers is None else args.workers)
    if not (1 <= budget <= 600) or int(cfg['dimension']) != 6 or len(seeds) != 10:
        raise ValueError('Stage 2 requires dimension 6, 10 seeds and budget 1..600')
    if not (1 <= workers <= 10):
        raise ValueError('Stage 2 workers must be 1..10')
    if float(stage['timeout_seconds']) <= 0:
        raise ValueError('stage2.timeout_seconds must be positive')
    if int(stage['botorch_initial_points']) < 1:
        raise ValueError('stage2.botorch_initial_points must be >= 1')
    optimizers = [load_winner('bbob'), load_winner('realworld'),
                  Algorithm('cmaes'), Algorithm('botorch',
                  options={'n_initial': int(stage['botorch_initial_points'])})]

    # Isolate results for each evaluation budget.
    out = RESULTS / 'stage2' / f'budget_{budget:04d}'
    if out.is_symlink():
        raise RuntimeError(f'Refusing to overwrite symlink: {out}')
    # Resolve both metrics before replacing existing benchmark output.
    metrics = {target: metric_for_target(cfg, target, RESULTS / 'calibration')
               for target in ('bbob', 'realworld')}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / 'experiment.json', dict(config=cfg, seeds=seeds, budget=budget,
              workers=workers, metrics=metrics,
              algorithms=[x.name for x in optimizers], machine=metadata()))

    summaries = []
    for target in ('bbob', 'realworld'):
        problem = from_config(cfg, target)
        optimum = known_optimum(problem) if target == 'bbob' else None
        all_runs, all_rows, statuses = [], [], []
        columns = ['method', 'problem', 'seed', 'evaluation', 'objective',
                   'best_so_far', 'status']
        if optimum is not None:
            columns.append('loss')
        columns.extend(f'x{j+1}' for j in range(int(cfg['dimension'])))
        for optimizer in optimizers:
            print(f'Evaluating {optimizer.name} on {target} '
                  f'({len(seeds)} seeds, budget={budget}, workers={workers})', flush=True)
            native_logs = out / 'ioh' if stage.get('ioh_logging', False) else None
            runs = evaluate_many(optimizer, problem, seeds, budget,
                float(stage['timeout_seconds']), workers,
                aocc_config=metrics[target], ioh_root=native_logs)
            all_runs.extend(runs)
            summary = summarize(runs, target)
            summaries.append(summary)
            print(f'  {summary["valid"]}/10 valid; final mean={summary["final_mean"]}; '
                  f'GP/Python warnings={summary["warning_count"]}', flush=True)
            for run in runs:
                all_rows.extend(rows(run, optimum))
                statuses.append(dict(problem=target, algorithm=optimizer.name,
                    seed=run.seed, status=run.status, error=run.error,
                    objective_calls=len(run.ys), seconds=run.seconds,
                    final_best=run.best, aocc=run.aocc,
                    warning_count=run.warning_count,
                    warnings=json.dumps(run.warnings, ensure_ascii=False)))
            # Persist each completed method/problem evaluation.
            save_csv(out / f'{target}_runs.csv', all_rows, columns)
            save_csv(out / f'{target}_statuses.csv', statuses)
            dump_json(out / f'{target}_results.json', [r.to_dict() for r in all_runs])
            save_csv(out / 'summary.csv', summaries)
            plot_stage2(all_runs, budget, out / 'figures' / f'{target}_convergence.png',
                        optimum=optimum)
            plot_stage2(all_runs, budget, out / 'figures' / f'{target}_normalized.png',
                        metric=metrics[target])
    print(f'Stage 2 complete. Results: {out} (check failed-run counts).')


if __name__ == '__main__':
    main()
