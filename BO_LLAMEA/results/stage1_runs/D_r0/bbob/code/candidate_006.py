import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness and WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 1.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Increase initial sampling to fill space better, but keep budget efficient
        n_init = max(self.dim + 2, min(self.budget // 4, 12))
        
        for i in range(self.budget):
            if i < n_init:
                # Use quasi-random/spread initialization for better initial coverage
                x = np.random.uniform(lb, ub)
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            # Adaptive Trust Region logic
            if i >= n_init:
                best_y = np.min(y_obs)
                if y < best_y:
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        try:
            self.gp.fit(X_norm, y)
        except:
            return np.random.uniform(lb, ub)
        
        y_min = np.min(y)
        best_x_norm = (X[np.argmin(y)] - lb) / (ub - lb)
        
        def ei(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            imp = y_min - mu
            z = imp / sigma
            return -(imp * norm.cdf(z) + sigma * norm.pdf(z)).flatten()

        # Multi-start acquisition optimization within the Trust Region
        best_acq = np.inf
        target_x = None
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.normal(0, self.tr_length / 3, self.dim), 0, 1)
            try:
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb