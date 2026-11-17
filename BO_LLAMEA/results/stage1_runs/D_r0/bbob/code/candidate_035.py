import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to allow different lengthscales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=2)
        self.best_y = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            if y < self.best_y:
                self.best_y = y
                self.best_x = x
            
            X_obs.append(x)
            y_obs.append(y)
            
            # Periodically re-fit hyperparameters as data grows
            if i > n_init and (i % 10 == 0):
                self.gp.n_restarts_optimizer = 1
            else:
                self.gp.n_restarts_optimizer = 0
                
        return float(self.best_y), np.array(self.best_x)

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        
        def acq(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-12)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        # Multi-start optimization of EI
        best_x_norm = None
        min_acq = 0
        for x0 in np.random.uniform(0, 1, size=(5, self.dim)):
            res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if res.fun < min_acq:
                min_acq = res.fun
                best_x_norm = res.x
        
        if best_x_norm is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(best_x_norm, 0, 1) * (ub - lb) + lb