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
        # Use ARD kernel for feature importance learning, add WhiteKernel for numerical stability
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False, n_restarts_optimizer=1)
        self.tr_size = 0.5 

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = x_norm.reshape(1, -1) * (ub - lb) + lb
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                y_best = np.min(self.y_hist)
                x_inc = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                tr_lb = np.maximum(x_inc_norm - self.tr_size, 0.0)
                tr_ub = np.minimum(x_inc_norm + self.tr_size, 1.0)
                
                res = minimize(
                    lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                    x0=np.clip(x_inc_norm, tr_lb, tr_ub),
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                if res.fun < -1e-7: self.tr_size = min(0.5, self.tr_size * 1.1)
                else: self.tr_size = max(0.02, self.tr_size * 0.5)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            if len(self.X_hist) >= n_init:
                # Log-transform target for better sensitivity to small values
                y_trans = np.log(np.array(self.y_hist) - np.min(self.y_hist) + 1e-9)
                X_trans = (np.array(self.X_hist) - lb) / (ub - lb)
                try:
                    self.gp.fit(X_trans, y_trans)
                except:
                    pass
                    
        return float(best_y), np.array(best_x)