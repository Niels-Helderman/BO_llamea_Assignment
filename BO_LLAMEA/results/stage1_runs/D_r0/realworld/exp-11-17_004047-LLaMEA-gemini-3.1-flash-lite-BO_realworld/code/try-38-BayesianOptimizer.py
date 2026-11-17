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
        # ARD Matern kernel allows independent length-scales for each dimension
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=3)
        self.tr_size = 0.5

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = (x_norm * (ub - lb) + lb).reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-12, None)
        diff = y_best - mu
        z = diff / sigma
        ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei.item()

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)
        
        n_init = max(2 * self.dim + 4, 16)
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                y_best = np.min(self.y_hist)
                x_inc = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                # Local search within trust region
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                res = minimize(
                    lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                    x0=x_inc_norm,
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Success-based trust region adjustment
                if res.fun < -1e-7:
                    self.tr_size = min(0.8, self.tr_size * 1.2)
                else:
                    self.tr_size = max(0.05, self.tr_size * 0.7)
            
            # Avoid sampling too close to existing points
            if len(self.X_hist) > 0:
                dist = np.linalg.norm((np.array(self.X_hist) - x) / (ub - lb), axis=1)
                if np.min(dist) < 1e-4:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Fit GP with scheduled re-estimation
            if len(self.X_hist) >= 8:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except Exception:
                    pass
                    
        return float(best_y), np.array(best_x)