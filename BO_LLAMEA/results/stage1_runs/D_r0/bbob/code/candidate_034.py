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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
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
        best_x, best_y = None, np.inf

        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs), best_x, lb, ub)
                x = x_norm * (ub - lb) + lb
            
            y = func(x)
            if y < best_y:
                best_y, best_x = y, x
                self.tr_length = min(self.max_tr, self.tr_length * 1.2)
            else:
                self.tr_length = max(self.min_tr, self.tr_length * 0.8)
            
            X_obs.append(x)
            y_obs.append(y)
            
            if self.tr_length <= self.min_tr * 1.1 and i > n_init + 5:
                self.tr_length = 0.5
                
        return float(best_y), np.array(best_x)

    def _propose(self, X, y, best_x, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        best_x_norm = (best_x - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        y_min = np.min(y)

        def acq(x_n):
            x_n = x_n.reshape(1, -1)
            mu, sigma = self.gp.predict(x_n, return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-9)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_x_acq = None
        max_ei = -np.inf
        
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if -res.fun > max_ei:
                    max_ei, best_x_acq = -res.fun, res.x
            except: continue
            
        return best_x_acq if best_x_acq is not None else np.random.uniform(0, 1, self.dim)