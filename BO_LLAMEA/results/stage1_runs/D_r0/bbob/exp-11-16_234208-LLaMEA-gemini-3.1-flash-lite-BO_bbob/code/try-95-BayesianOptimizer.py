import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use ARD kernel: each dimension has its own length scale
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=False, n_restarts_optimizer=2)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space coverage in high dimensions
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        # Transform X to [0, 1]^d and target to log space to handle non-negative BBOB ranges
        X_norm = (X - lb) / (ub - lb)
        y_trans = np.log1p(y - np.min(y) + 1e-6)
        
        # Refit GP occasionally or every step; limit hyperparameter opt iterations
        self.gp.fit(X_norm, y_trans)
        
        y_min_trans = np.min(y_trans)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-10, None)
            z = (y_min_trans - mu) / sigma
            ei = (y_min_trans - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        # Multi-start optimization of acquisition function
        best_val = np.inf
        best_x = np.random.rand(self.dim)
        
        starts = np.random.rand(5, self.dim)
        for x0 in starts:
            res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
        return np.clip(best_x, 0, 1) * (ub - lb) + lb