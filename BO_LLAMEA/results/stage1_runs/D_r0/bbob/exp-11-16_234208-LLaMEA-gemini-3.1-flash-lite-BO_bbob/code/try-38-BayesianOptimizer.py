import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to capture different feature sensitivities
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        self.tr_length = 0.5
        self.best_y = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            if y < self.best_y:
                self.best_y = y
                self.best_x = x
                self.tr_length = min(0.5, self.tr_length * 1.2)
            else:
                self.tr_length = max(0.01, self.tr_length * 0.9)
                
        return float(self.best_y), np.array(self.best_x)

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y_vec = np.array(y)
        
        # Fit GP periodically or always; ARD captures anisotropy
        self.gp.fit(X_norm, y_vec)
        
        y_min = np.min(y_vec)
        best_x_norm = X_norm[np.argmin(y_vec)]
        
        def acquisition(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                diff = y_min - mu
                z = diff / (sigma + 1e-9)
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_acq_x = None
        best_val = np.inf
        
        # Local search constrained to trust region around current best
        for _ in range(8):
            x0 = np.clip(best_x_norm + np.random.uniform(-self.tr_length, self.tr_length, self.dim), 0, 1)
            try:
                res = minimize(acquisition, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_acq_x = res.x
            except:
                continue
        
        if best_acq_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(best_acq_x, 0, 1) * (ub - lb) + lb