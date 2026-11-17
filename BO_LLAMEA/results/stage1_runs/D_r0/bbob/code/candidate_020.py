import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to capture different scales per dimension
        self.kernel = Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=1)
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(lb, ub)
            
            y = func(x)
            self.X_obs.append(x)
            self.y_obs.append(y)
            
        best_idx = np.argmin(self.y_obs)
        return float(self.y_obs[best_idx]), np.array(self.X_obs[best_idx])

    def _propose(self, lb, ub):
        X = (np.array(self.X_obs) - lb) / (ub - lb)
        y = np.array(self.y_obs)
        self.gp.fit(X, y)
        
        y_min = np.min(y)
        
        def acq(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-12)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_val = np.inf
        best_x = None
        
        # Multi-start optimization of EI
        for _ in range(8):
            x0 = np.random.rand(self.dim)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except: continue
            
        if best_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(best_x, 0, 1) * (ub - lb) + lb