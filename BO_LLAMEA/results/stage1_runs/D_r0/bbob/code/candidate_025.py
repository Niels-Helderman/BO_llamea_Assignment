import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Use a single regressor; re-fit periodically
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_x_norm = None
        best_y = np.inf
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs), best_x_norm)
                x = x_norm * (ub - lb) + lb
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y = y
                best_x_norm = (x - lb) / (ub - lb)
                if i >= n_init:
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
            elif i >= n_init:
                self.tr_length = max(self.min_tr, self.tr_length * 0.8)
                if self.tr_length <= self.min_tr:
                    self.tr_length = 0.5 # Restart strategy
            
        return float(best_y), np.array(X_obs[np.argmin(y_obs)])

    def _propose(self, X, y, best_x_norm):
        X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_x = best_x_norm
        best_val = np.inf
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.uniform(-self.tr_length, self.tr_length, self.dim), 0, 1)
            res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x