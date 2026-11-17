import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use ARD (Automatic Relevance Determination) with anisotropic Matern kernel
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2)
        self.tr_length = 0.4
        self.y_shift = 0.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 1, self.budget // 5)
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
            
            # Adaptive Trust Region logic
            if i >= n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(0.8, self.tr_length * 1.1)
                else:
                    self.tr_length = max(0.05, self.tr_length * 0.9)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        # Log-transform target to handle potentially non-Gaussian objective distributions
        y_arr = np.array(y)
        self.y_shift = np.min(y_arr) - 1e-3
        y_log = np.log(y_arr - self.y_shift + 1e-6)
        
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y_log)
        
        y_min_log = np.min(y_log)
        best_x_norm = X_norm[np.argmin(y_log)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-10, None)
            z = (y_min_log - mu) / sigma
            return -( (y_min_log - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Multi-start optimization with local bias (Trust Region)
        best_acq_val, target_x = np.inf, None
        starts = [best_x_norm] + [np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1) for _ in range(8)]
        starts.append(np.random.rand(self.dim))
        
        for x0 in starts:
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                if res.fun < best_acq_val:
                    best_acq_val, target_x = res.fun, res.x
            except: continue
            
        return np.clip(target_x, 0, 1) * (ub - lb) + lb if target_x is not None else np.random.uniform(lb, ub)