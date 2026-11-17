import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a flexible Matern kernel with a fixed small noise floor for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        # Use a larger initial sampling set for better space coverage in 6D
        self.n_init = max(10, 2 * self.dim + 2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < self.n_init:
                # Simple random initialization in [0, 1] mapped to bounds
                x = np.random.uniform(lb, ub)
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)

        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        # Normalize inputs to [0,1] for better GP conditioning
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        
        try:
            self.gp.fit(X_norm, y)
        except:
            return np.random.uniform(lb, ub)
            
        y_min = np.min(y)
        
        def ei(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Multi-start acquisition optimization
        best_acq = -np.inf
        target_x = None
        for _ in range(8):
            x0 = np.random.uniform(0, 1, self.dim)
            res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if -res.fun > best_acq:
                best_acq = -res.fun
                target_x = res.x
        
        if target_x is None:
            return np.random.uniform(lb, ub)
        
        return target_x * (ub - lb) + lb