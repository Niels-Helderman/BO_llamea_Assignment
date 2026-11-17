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
        self.tr_center = None
        self.tr_length = 0.5
        self.min_tr = 0.01
        self.max_tr = 0.5
        self.success_count = 0
        self.fail_count = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(self.dim + 2, min(self.budget // 4, 15))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        best_x = None
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y, best_x = y, x
                if i >= n_init:
                    self.success_count += 1
                    self.fail_count = 0
                    if self.success_count >= 2:
                        self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                        self.success_count = 0
            elif i >= n_init:
                self.fail_count += 1
                self.success_count = 0
                if self.fail_count >= 3:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.5)
                    self.fail_count = 0
            
            if self.tr_length <= self.min_tr + 1e-4:
                self.tr_center = None
                self.tr_length = 0.5
                
        return float(best_y), best_x

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        best_idx = np.argmin(y)
        self.tr_center = X_norm[best_idx]
        y_min = y[best_idx]
        
        def acq(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-10)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_acq_x = None
        best_acq_val = -np.inf
        
        # Local search in trust region
        for _ in range(8):
            x0 = np.clip(self.tr_center + np.random.uniform(-self.tr_length, self.tr_length, self.dim), 0, 1)
            res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if -res.fun > best_acq_val:
                best_acq_val = -res.fun
                best_acq_x = res.x
        
        if best_acq_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(best_acq_x, 0, 1) * (ub - lb) + lb