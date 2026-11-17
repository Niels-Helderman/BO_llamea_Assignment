import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds='fixed') + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 1.0

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, 8)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < n_init:
                x_norm = sampler.random(1).flatten()
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs), best_x_norm)
            
            x = x_norm * (ub - lb) + lb
            y = func(x)
            
            X_obs.append(x_norm)
            y_obs.append(y)
            
            if y < best_y:
                best_y, best_x, best_x_norm = y, x, x_norm
                if i >= n_init: self.tr_length = min(self.max_tr, self.tr_length * 1.2)
            elif i >= n_init:
                self.tr_length *= 0.8
                if self.tr_length < self.min_tr:
                    self.tr_length = 0.5 # Reset TR
                    
        return float(best_y), best_x

    def _propose(self, X, y, best_x_norm):
        y = np.array(y)
        self.gp.fit(X, y)
        y_min = np.min(y)
        
        def acq(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-10)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_acq_val = -np.inf
        target_x = None
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
            if -res.fun > best_acq_val:
                best_acq_val, target_x = -res.fun, res.x
        
        return target_x if target_x is not None else np.random.rand(self.dim)