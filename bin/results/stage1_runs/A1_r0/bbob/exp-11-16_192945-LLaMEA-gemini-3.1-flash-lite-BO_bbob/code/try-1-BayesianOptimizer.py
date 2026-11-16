import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_sample, y_sample = [], []
        
        # Initialization: Use a small fraction of the budget for Latin Hypercube-like spread
        n_init = min(10, self.budget // 5)
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                x = self._propose_next(X_sample, y_sample, lb, ub)
            
            y = func(x)
            X_sample.append(x)
            y_sample.append(y)
            
            if len(X_sample) >= 3:
                # Periodic refitting: keep frequency low to respect budget
                self.gp.fit(np.array(X_sample), np.array(y_sample))
        
        best_idx = np.argmin(y_sample)
        return float(y_sample[best_idx]), np.array(X_sample[best_idx])

    def _propose_next(self, X, y, lb, ub):
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, std = self.gp.predict(x, return_std=True)
            y_min = np.min(y)
            # Expected Improvement
            diff = y_min - mu
            z = diff / (std + 1e-9)
            from scipy.stats import norm
            ei = diff * norm.cdf(z) + std * norm.pdf(z)
            return -ei[0] if std > 1e-9 else 0

        # Multi-start gradient-based optimization of the acquisition function
        best_x = None
        best_acq = np.inf
        for _ in range(8):
            x0 = np.random.uniform(lb, ub)
            res = minimize(acquisition, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        return best_x if best_x is not None else np.random.uniform(lb, ub)