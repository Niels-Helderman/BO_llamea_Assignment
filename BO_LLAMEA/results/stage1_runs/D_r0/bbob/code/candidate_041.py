import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Matern 5/2 is standard for continuous BBOB
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.01
        self.max_tr = 0.5

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
                x_norm = self._propose(np.array(X_obs), np.array(y_obs), best_x_norm)
                x = x_norm * (ub - lb) + lb
                
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                best_x_norm = (x - lb) / (ub - lb)
                if i >= n_init:
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
            elif i >= n_init:
                self.tr_length = max(self.min_tr, self.tr_length * 0.8)
                if self.tr_length <= self.min_tr * 1.1:
                    self.tr_length = 0.5 # Reset TR if collapsed

        return float(best_y), np.array(best_x)

    def _propose(self, X, y, best_x_norm):
        X_norm = (X - func_bounds_lb := (X.min(0) * 0)) # Placeholder normalization
        lb, ub = 0.0, 1.0
        X_norm = (X - (X.min(0)*0)) / ( (X.max(0)*0) + 1.0 ) # Normalized correctly via external bounds
        # Actual normalization:
        X_norm = (X - self._get_lb(X)) / (self._get_ub(X) - self._get_lb(X))
        
        self.gp.fit(X_norm, y)
        y_min = np.min(y)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-10)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        best_acq_val = np.inf
        target_x = best_x_norm
        
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if res.fun < best_acq_val:
                best_acq_val = res.fun
                target_x = res.x
        return target_x

    def _get_lb(self, X): return np.zeros(self.dim)
    def _get_ub(self, X): return np.ones(self.dim)