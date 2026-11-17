import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 1.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Use Sobol sequence for better space-filling initialization
        n_init = min(self.budget, 2 * self.dim + 4)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sobol.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            # Adaptive Trust Region logic
            if i > n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
            
            if self.tr_length <= self.min_tr + 1e-4:
                self.tr_length = 0.5

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
        
        def acq(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-9)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_acq = -np.inf
        target_x = None
        # Multi-start optimization within the trust region
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.uniform(-self.tr_length, self.tr_length, self.dim), 0, 1)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        return target_x * (ub - lb) + lb