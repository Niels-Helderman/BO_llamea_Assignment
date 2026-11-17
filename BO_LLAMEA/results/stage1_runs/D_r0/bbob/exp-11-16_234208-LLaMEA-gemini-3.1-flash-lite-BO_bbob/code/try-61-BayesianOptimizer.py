import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.02
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y = y
                self.tr_length = min(self.max_tr, self.tr_length * 1.1)
            else:
                self.tr_length = max(self.min_tr, self.tr_length * 0.9)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y_min = np.min(y)
        self.gp.fit(X_norm, y)
        
        def ei(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))

        best_x = None
        best_val = 0.0
        
        # Multi-start acquisition optimization using L-BFGS-B
        # Focus on trust region around current best
        current_best_x = X_norm[np.argmin(y)]
        seeds = [current_best_x + np.random.normal(0, self.tr_length, self.dim) for _ in range(8)]
        seeds.append(np.random.rand(self.dim))
        
        for x0 in seeds:
            x0 = np.clip(x0, 0, 1)
            try:
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            best_x = np.random.rand(self.dim)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb