"""Small shared helpers: configuration, reproducibility, storage and plots."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'experiment.yaml'
RESULTS = ROOT / 'results'


def load_config(path=DEFAULT_CONFIG) -> dict:
    """Input: YAML path. Output: configuration dictionary."""
    import yaml
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Expected YAML mapping')
    return cfg


def group_seeds(cfg: dict) -> list[int]:
    """Input: group ID from YAML. Output: exactly 10 fixed assignment seeds."""
    g = int(cfg['group_id'])
    if not 1 <= g <= 14:
        raise ValueError('Group ID must be 1..14')
    return list(range(10 * g, 10 * g + 10))


def metadata() -> dict:
    """Output: versions/platform recorded with every real experiment."""
    packages = {}
    for pkg in ('llamea', 'ioh', 'Practical_Problems', 'numpy', 'scipy', 'cma',
                'botorch', 'gpytorch', 'torch', 'matplotlib', 'PyYAML'):
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = 'not installed'
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages}


def dump_json(path, obj):
    """Input: path and JSON-serializable data. Writes formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so power loss/interruption cannot leave half a JSON file.
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save_csv(path, data: list[dict], columns: list[str] | None = None):
    """Input: rows and optional headers. Output: CSV including header if empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or list(dict.fromkeys(k for row in data for k in row))
    if not fields:
        raise ValueError('Specify columns for an empty CSV')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def plot_one(result, path):
    """Input: one result. Output: best-so-far + observed objective PNG."""
    if not result.ys:
        return
    from .performance import best_so_far
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(1, len(result.ys) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, best_so_far(result.ys), label='Best so far')
    ax.scatter(x, result.ys, s=10, alpha=.35, label='Observed objective')
    ax.set(xlabel='Objective evaluations', ylabel='Objective', title=f'{result.algorithm}: seed {result.seed}')
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fitness(records, path):
    """AOCC on a shared 0..1 scale; invalid candidates receive zero fitness."""
    if not records:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fitness = np.asarray([r['fitness'] for r in records], dtype=float)
    valid = np.asarray([r['valid'] for r in records], dtype=bool)
    steps = np.arange(1, len(records) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if valid.any():
        ax.scatter(steps[valid], fitness[valid], label='Valid candidate', s=25)
        # Invalid candidates receive zero fitness and cannot improve the best-so-far AOCC.
        best = np.maximum.accumulate(np.where(valid, fitness, -np.inf))
        best[~np.isfinite(best)] = np.nan
        ax.plot(steps, best, label='Best-so-far AOCC')
    if (~valid).any():
        ax.scatter(steps[~valid], np.zeros((~valid).sum()), c='crimson',
                   marker='x', label='Invalid (AOCC fitness: 0)')
    ax.set(xlabel='Candidates evaluated', ylabel='Mean normalized AOCC (10 seeds)',
           ylim=(-.035, 1.03), title='LLaMEA search (higher is better)')
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stage2(results, budget, path, optimum=None, metric=None):
    """Mean best-so-far +/- SE, either raw loss/objective or relative gap.

    Real-world *raw* plots stay in objective units; BBOB raw plots show
    instance-specific loss. Normalized plots use the fixed per-target AOCC
    reference and a common log y-axis. With ``metric['scale']`` they show the
    existing relative gap; without it they show the paper's unscaled
    best-so-far precision.

    Failed runs are excluded from means and standard errors. Their counts are
    retained in the status and summary outputs.
    """
    from .performance import best_so_far
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    any_good = False
    for name in sorted({r.algorithm for r in results}):
        runs = [r for r in results if r.algorithm == name and r.status == 'ok']
        if not runs:
            continue
        any_good = True
        curves = np.stack([best_so_far(r.ys, budget) for r in runs])
        if metric is not None:
            curves = curves - float(metric['reference'])
            if metric.get('scale') is not None:
                curves = curves / float(metric['scale'])
            curves = np.maximum(curves, 1e-12)
        elif optimum is not None:
            curves = curves - optimum
        avg = curves.mean(axis=0)
        se = (curves.std(axis=0, ddof=1) / np.sqrt(len(runs))
              if len(runs) > 1 else np.zeros(budget))
        x = np.arange(1, budget + 1)
        line, = ax.plot(x, avg, label=f'{name} (n={len(runs)})')
        lower = avg - se
        upper = avg + se
        if metric is not None:
            lower = np.maximum(lower, 1e-12)
        ax.fill_between(x, lower, upper, color=line.get_color(), alpha=.15)
    if metric is not None:
        ax.set_yscale('log')
        # Identical axis limits across targets: the same AOCC log-error window.
        ax.set_ylim(10.0 ** float(metric['lower']), 10.0 ** float(metric['upper']))
        if metric.get('scale') is None:
            ax.set_ylabel('Best-so-far precision: f - reference')
            ax.set_title('Paper AOCC precision view (lower is better)')
        else:
            ax.set_ylabel('Relative best-so-far gap: (f - reference) / scale')
            ax.set_title('Normalized cross-problem view (lower is better)')
    else:
        ax.set_ylabel('Best-so-far loss (f - f*)' if optimum is not None
                      else 'Best-so-far objective')
        ax.set_title('Mean convergence +/- standard error')
    ax.set_xlabel('Actual objective evaluations (padded best-so-far only)')
    if any_good:
        ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def seed_search(seed: int):
    """Input: outer-loop seed. Seeds Python/NumPy for ES-side choices."""
    random.seed(seed)
    np.random.seed(seed)
