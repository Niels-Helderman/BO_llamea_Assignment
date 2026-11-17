import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import boxcox, inv_boxcox, boxcox1p

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # ARD Matern kernel to handle independent dimension scales
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False, n_restarts_optimizer=3)
        self.tr_size = 0.5
        self.lambda_bc = 0.0 # Placeholder for Box-Cox parameter

    def _warp_y(self, y):
        # Shift and scale to [1, inf) for boxcox1p
        y_min = np.min(self.y_hist)
        return boxcox1p(y - y_min, 0.0)

    def _get_ei(self, x_norm, gp, y_best_warped, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best_warped - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

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
                y_warped = self._warp_y(np.array(self.y_hist))
                y_best_warped = np.min(y_warped)
                x_inc = self.X_hist[np.argmin(y_warped)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                res = minimize(
                    lambda x_n: self._get_ei(x_n, self.gp, y_best_warped, lb, ub),
                    x0=x_inc_norm,
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                if res.fun < -1e-5:
                    self.tr_size = min(0.5, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.01, self.tr_size * 0.5)
            
            x = np.clip(x, lb, ub)
            
            if len(self.X_hist) > 0 and np.min(np.linalg.norm(np.array(self.X_hist) - x, axis=1)) < 1e-7:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            if len(self.X_hist) >= 6:
                try:
                    y_w = self._warp_y(np.array(self.y_hist))
                    self.gp.fit(np.array(self.X_hist), y_w)
                except:
                    pass
                    
        return float(best_y), np.array(best_x)