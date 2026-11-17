import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.02
        self.max_tr = 0.5
        self.success_counter = 0
        self.failure_counter = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 4)
        
        X_obs, y_obs = [], []
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
            if i >= n_init:
                if y < np.min(y_obs[:-1]):
                    self.success_counter += 1
                    self.failure_counter = 0
                    if self.success_counter >= 3:
                        self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                        self.success_counter = 0
                else:
                    self.failure_counter += 1
                    self.success_counter = 0
                    if self.failure_counter >= 5:
                        self.tr_length = max(self.min_tr, self.tr_length * 0.5)
                        self.failure_counter = 0
            
            if self.tr_length <= self.min_tr and i > n_init + 10:
                self.tr_length = 0.5
                
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y_min = np.min(y)
        try:
            self.gp.fit(X_norm, y)
        except:
            return np.random.uniform(lb, ub)
            
        best_x_norm = X_norm[np.argmin(y)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        best_x = None
        min_acq = np.inf
        
        candidates = [np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)]
        candidates.append(np.random.rand(self.dim))
        
        for x0 in candidates:
            res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
            if res.fun < min_acq:
                min_acq = res.fun
                best_x = res.x
                
        return np.clip(best_x, 0, 1) * (ub - lb) + lb