import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel setup with a small constant noise floor
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(lb, ub)
            
            y = float(func(x))
            self.X_obs.append(x)
            self.y_obs.append(y)
            
        best_idx = np.argmin(self.y_obs)
        return float(self.y_obs[best_idx]), np.array(self.X_obs[best_idx])

    def _propose(self, lb, ub):
        X = (np.array(self.X_obs) - lb) / (ub - lb)
        y = np.array(self.y_obs)
        
        # Fit GP with occasional hyperparameter updates
        try:
            self.gp.fit(X, y)
        except:
            return np.random.uniform(lb, ub)
            
        y_min = np.min(y)
        
        def neg_ei(x_n):
            x_n = x_n.reshape(1, -1)
            mu, sigma = self.gp.predict(x_n, return_std=True)
            sigma = np.clip(sigma, 1e-12, None)
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        # Multi-start optimization of EI
        best_acq = -np.inf
        target_x = None
        starts = np.vstack([X[np.argmin(y)], np.random.rand(10, self.dim)])
        
        for x0 in starts:
            try:
                res = minimize(neg_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(target_x, 0, 1) * (ub - lb) + lb