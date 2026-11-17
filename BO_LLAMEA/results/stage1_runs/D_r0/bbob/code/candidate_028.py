import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small white noise for deterministic stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0, n_restarts_optimizer=0)
        self.rng = np.random.RandomState(42)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                # Alternate between GP-guided search and random exploration to escape local optima
                if i % 8 == 0:
                    x = self.rng.uniform(lb, ub)
                else:
                    x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        
        # Periodic refitting or limit iterations for efficiency
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        
        def ei(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Multi-start acquisition optimization
        best_acq = np.inf
        target_x = None
        
        starts = np.vstack([X_norm[np.argmin(y)], self.rng.rand(5, self.dim)])
        for x0 in starts:
            try:
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-5)
                if res.fun < best_acq:
                    best_acq = res.fun
                    target_x = res.x
            except:
                continue
        
        if target_x is None:
            return self.rng.uniform(lb, ub)
        
        return np.clip(target_x, 0, 1) * (ub - lb) + lb