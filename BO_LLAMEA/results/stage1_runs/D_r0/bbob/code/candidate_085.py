import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel for smooth but potentially sharp functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_center = None
        self.tr_length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, i)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            if y < best_y:
                best_y = y
                self.tr_center = (x - lb) / (ub - lb)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, i):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        # Adaptive Beta for UCB: explore early, exploit later
        beta = 2.0 * np.log(i + 1)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        # Multi-start optimization with focus on the current Trust Region
        best_acq_val = np.inf
        target_x = None
        
        candidates = []
        # Local search around best observed
        for _ in range(10):
            candidates.append(np.clip(self.tr_center + np.random.normal(0, self.tr_length, self.dim), 0, 1))
        # Global search
        candidates.append(np.random.rand(self.dim))
        
        for x0 in candidates:
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        
        # Update TR: shrink if stuck, expand if useful (basic logic)
        self.tr_length = np.clip(self.tr_length * 0.99, 0.01, 0.5)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb