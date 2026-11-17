import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.02
        self.max_tr = 1.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        best_x = None

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
                if i >= n_init: self.tr_length = min(self.max_tr, self.tr_length * 1.1)
            elif i >= n_init:
                self.tr_length = max(self.min_tr, self.tr_length * 0.9)
                if self.tr_length <= self.min_tr * 1.1: self.tr_length = 0.5 # Restart TR
            
        return float(best_y), np.array(best_x)

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        y_min = np.min(y)
        incumbent_x = X_norm[np.argmin(y)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))

        best_acq_val = np.inf
        target_x = None
        
        # Local search around incumbent + global random candidates
        candidates = [np.clip(incumbent_x + np.random.normal(0, self.tr_length, self.dim), 0, 1) for _ in range(8)]
        candidates.append(np.random.rand(self.dim))
        
        for x0 in candidates:
            try:
                res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq_val:
                    best_acq_val, target_x = res.fun, res.x
            except: continue
            
        if target_x is None: return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb