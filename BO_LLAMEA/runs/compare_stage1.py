"""Compare Stage 1 configurations: python compare_stage1.py results/stage1_runs bbob"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# Rough design fingerprint from source code (a proxy, not ground truth).
KEYS = dict(rf=r'RandomForest|ExtraTrees', thompson=r'thompson|sample_y|rsample',
            ucb=r'\bucb\b|\blcb\b|beta', trust=r'trust|radius', cma=r'cma|differential_evolution',
            sobol=r'Sobol|LatinHypercube', botorch=r'botorch', matern=r'Matern')

def load(folder):
    recs = []
    for f in sorted((folder / 'code').glob('candidate_*.json')):
        r = json.loads(f.read_text())
        code = (folder / r['code']).read_text() if (folder / r['code']).exists() else ''
        sig = frozenset(k for k, p in KEYS.items() if re.search(p, code, re.I))
        recs.append((r['fitness'], r['valid'], sig))
    return recs

root, target = Path(sys.argv[1]), sys.argv[2]
runs = defaultdict(list)
for d in sorted(root.glob(f'*_r*/{target}')):
    runs[d.parent.name.rsplit('_r', 1)[0]].append(load(d))
for tag, reps in sorted(runs.items()):
    n = min(len(r) for r in reps)
    best = np.array([np.maximum.accumulate([f for f, _, _ in r[:n]]) for r in reps])
    marks = [i for i in (5, 10, 20, 30, 40) if i <= n]
    curve = ' '.join(f'{i}:{best[:, i-1].mean():.3f}' for i in marks)
    valid = np.mean([v for r in reps for _, v, _ in r[:n]])
    designs = np.mean([len({s for _, _, s in r[:n]}) for r in reps])
    print(f'{tag:4s} reps={len(reps)} n={n} best-so-far[{curve}] '
          f'final={best[:, -1].mean():.3f}±{best[:, -1].std():.3f} valid={valid:.0%} distinct_designs={designs:.1f}')