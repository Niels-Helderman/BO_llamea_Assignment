import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.05
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Increase initial exploration to cover space better in first 30% of budget
        n_init = max(2 * self.dim + 1, self.budget // 4)
        
        for i in range(self.budget):
            if i < n_init:
                # Sobol-like spread using jittered uniform
                x = np.random.uniform(lb, ub)
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            # Simple adaptive TR update based on improvement
            if i > n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        
        # Periodic GP fit
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        best_x_norm = (X[np.argmin(y)] - lb) / (ub - lb)
        
        def objective(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        # Multi-start acquisition optimization within TR
        best_acq = -np.inf
        target_x = None
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.normal(0, self.tr_length / 2, self.dim), 0, 1)
            try:
                res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb