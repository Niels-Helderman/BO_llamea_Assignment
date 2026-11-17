"""Reference optimizers with the shared (budget, dim), __call__(f) interface.

BoTorch and CMA-ES hyperparameters are defined here for repeatable benchmarks.
"""
import numpy as np


class RandomSearch:
    """Uniform random search for internal evaluation checks."""
    def __init__(self, budget: int, dim: int):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        lb, ub = np.asarray(func.bounds.lb), np.asarray(func.bounds.ub)
        best_y, best_x = float('inf'), None
        for _ in range(self.budget):
            x = np.random.uniform(lb, ub, self.dim)
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x.copy()
        return best_y, best_x


class CMAES:
    """Use pycma ask/tell; no extra calls to finish an incomplete final batch."""
    def __init__(self, budget: int, dim: int):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        import cma
        lb, ub = np.asarray(func.bounds.lb), np.asarray(func.bounds.ub)
        es = cma.CMAEvolutionStrategy(((lb + ub) / 2).tolist(),
             0.25 * float(np.mean(ub - lb)), {
                 'bounds': [lb.tolist(), ub.tolist()],
                 'seed': int(np.random.randint(1, 2**31 - 1)), 'verbose': -9})
        count, best = 0, (float('inf'), None)
        while count < self.budget and not es.stop():
            proposed = es.ask()
            n = min(len(proposed), self.budget - count)
            points = [np.clip(np.asarray(x, dtype=float), lb, ub) for x in proposed[:n]]
            values = []
            for x in points:
                y = float(func(x))
                values.append(y)
                count += 1
                if y < best[0]:
                    best = (y, x.copy())
            if n == len(proposed):
                es.tell([x.tolist() for x in points], values)
        return best


class BoTorchBO:
    """CPU SingleTaskGP + log-EI, minimizing objective via negative observations."""
    def __init__(self, budget: int, dim: int, n_initial: int = 12):
        self.budget, self.dim, self.n_initial = budget, dim, n_initial

    def __call__(self, func):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        dtype = torch.double
        lb = torch.as_tensor(func.bounds.lb, dtype=dtype)
        ub = torch.as_tensor(func.bounds.ub, dtype=dtype)
        bounds = torch.stack([lb, ub])
        engine = torch.quasirandom.SobolEngine(self.dim, scramble=True)
        X, Y = [], []
        def observe(point):
            # Keep one objective evaluation per call, including initialization.
            y = float(func(point.detach().cpu().numpy()))
            X.append(point.detach().clone())
            Y.append(y)
        for p in lb + (ub - lb) * engine.draw(min(self.n_initial, self.budget)).to(dtype):
            observe(p)
        while len(Y) < self.budget:
            train_x = torch.stack(X)
            train_y = -torch.tensor(Y, dtype=dtype).unsqueeze(-1)
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acquisition = LogExpectedImprovement(model, best_f=float(train_y.max()))
            x, _ = optimize_acqf(acquisition, bounds=bounds, q=1,
                                 num_restarts=5, raw_samples=64)
            observe(x[0])
        idx = int(np.argmin(Y))
        return float(Y[idx]), X[idx].detach().cpu().numpy()


def get_baseline(name: str):
    """Input: baseline name. Output: optimizer class with shared interface."""
    choices = {'random': RandomSearch, 'cmaes': CMAES, 'botorch': BoTorchBO}
    if name not in choices:
        raise ValueError(f'Unknown baseline: {name}')
    return choices[name]
