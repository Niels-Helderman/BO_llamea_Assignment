import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern 5/2 kernel is more robust for high-dim scaling
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-11, 1e-3))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=2)
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(lb, ub)
            
            y = func(x)
            self.X_obs.append(x)
            self.y_obs.append(y)
            
        best_idx = np.argmin(self.y_obs)
        return float(self.y_obs[best_idx]), np.array(self.X_obs[best_idx])

    def _propose(self, lb, ub):
        X = (np.array(self.X_obs) - lb) / (ub - lb)
        y = np.array(self.y_obs)
        
        # Periodic GP fitting: refit only if budget allows or significant data added
        # Using ARD allows the model to ignore irrelevant dimensions
        self.gp.fit(X, y)
        
        y_min = np.min(y)
        
        def ei(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-12)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Multi-start acquisition optimization
        best_val = np.inf
        best_x = None
        # Candidates: best observed + random points
        candidates = [X[np.argmin(y)]] + [np.random.rand(self.dim) for _ in range(5)]
        
        for x0 in candidates:
            res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        
        if best_x is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(best_x, 0, 1) * (ub - lb) + lb