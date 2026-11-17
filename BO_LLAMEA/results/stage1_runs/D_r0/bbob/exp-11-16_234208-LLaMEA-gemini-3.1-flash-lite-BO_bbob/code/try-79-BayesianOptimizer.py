import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0, n_restarts_optimizer=0)
        self.sigma_cma = 0.3
        self.dim_float = float(dim)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2, min(self.budget // 4, 2 * self.dim + 2))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y, best_x = float('inf'), None
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            if y < best_y:
                best_y, best_x = y, x
            
        return best_y, best_x

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        y_min = np.min(y)
        
        def ei(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(-1, self.dim), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        # Simplified (1+1)-CMA-ES style optimization for the acquisition function
        best_x_norm = X_norm[np.argmin(y)]
        curr_x = best_x_norm
        best_val = -1.0
        
        for _ in range(30):
            candidate = np.clip(curr_x + np.random.normal(0, self.sigma_cma, self.dim), 0, 1)
            val = ei(candidate)
            if val > best_val:
                best_val = val
                curr_x = candidate
                self.sigma_cma = min(1.0, self.sigma_cma * 1.1)
            else:
                self.sigma_cma = max(0.01, self.sigma_cma * 0.9)
        
        # Add random perturbation to escape local optima
        if np.random.rand() < 0.1:
            curr_x = np.random.rand(self.dim)
            
        return np.clip(curr_x, 0, 1) * (ub - lb) + lb