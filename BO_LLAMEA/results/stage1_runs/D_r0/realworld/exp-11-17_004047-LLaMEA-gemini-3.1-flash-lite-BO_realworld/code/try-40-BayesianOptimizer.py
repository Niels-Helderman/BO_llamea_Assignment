import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # Robust kernel: Matern 5/2 handles medium smoothness, WhiteKernel handles noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=1)
        self.tr_size = 0.5
        self.best_y = np.inf

    def _get_ei(self, x_norm, lb, ub):
        x = (x_norm * (ub - lb) + lb).reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = self.best_y - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, max(2 * self.dim + 1, 10))
        
        for i in range(self.budget):
            if i < n_init:
                # Latin Hypercube-like spread using Sobol-like perturbation
                x = lb + (ub - lb) * ((np.arange(n_init) + 0.5) / n_init) # simplified for logic
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                x_inc = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                # Optimize EI in trust region
                res = minimize(
                    lambda x_n: self._get_ei(x_n, lb, ub),
                    x0=x_inc_norm,
                    bounds=[(max(0, x_inc_norm[j]-self.tr_size), min(1, x_inc_norm[j]+self.tr_size)) for j in range(self.dim)],
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Update TR: if better than best_y, expand/maintain, else shrink
                if res.fun < -1e-9:
                    self.tr_size = min(0.5, self.tr_size * 1.2)
                else:
                    self.tr_size *= 0.7
                
                if self.tr_size < 0.01:
                    self.tr_size = 0.5 # Reset TR if collapsed

            x = np.clip(x, lb, ub)
            # Prevent duplicate queries
            if len(self.X_hist) > 0 and np.min(np.linalg.norm(np.array(self.X_hist) - x, axis=1)) < 1e-7:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < self.best_y:
                self.best_y = y
            
            # Refit GP periodically
            if len(self.X_hist) >= 5:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        best_idx = np.argmin(self.y_hist)
        return float(self.y_hist[best_idx]), np.array(self.X_hist[best_idx])