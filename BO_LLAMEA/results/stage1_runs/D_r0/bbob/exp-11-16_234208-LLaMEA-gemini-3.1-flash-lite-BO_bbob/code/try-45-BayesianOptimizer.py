import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel for BBOB: Matern 5/2 with learned length scale
        self.kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=1)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial design size for 6D problem
        n_init = min(self.budget // 3, 2 * self.dim + 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        init_points = sampler.random(n_init)
        
        X_obs = []
        y_obs = []
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs))
                x = np.clip(x_norm, 0, 1) * (ub - lb) + lb
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            # Adaptive TR logic
            if i >= n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
                    
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y):
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-9)
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        def ei(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-12)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        best_acq = np.inf
        target_x = None
        # Efficient multi-start within Trust Region
        for _ in range(8):
            start_x = np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            res = minimize(ei, start_x, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if res.fun < best_acq:
                best_acq = res.fun
                target_x = res.x
        
        return target_x if target_x is not None else np.random.uniform(0, 1, self.dim)