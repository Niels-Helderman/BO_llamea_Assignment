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
        # Robust kernel: Matern 5/2 is standard for BBO, WhiteKernel handles noisy evaluations
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        self.tr_size = 0.5 

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb).clip(lb, ub)
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
        
        # Initial phase: Use a smaller, more effective LHS initialization to start GP fitting faster
        n_init = max(2 * self.dim + 2, 12)
        
        for i in range(self.budget):
            if i < n_init:
                # Latin Hypercube-like spacing for robust initialization
                x = lb + (ub - lb) * (np.random.rand(self.dim) + np.arange(self.dim)) / (n_init * 0.5)
                x = np.clip(x, lb, ub)
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                # Trust Region to prevent over-exploitation and premature convergence
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                res = minimize(
                    lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                    x0=np.clip(x_inc_norm, tr_lb, tr_ub),
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # TR adjustment logic
                if res.fun < -1e-6:
                    self.tr_size = min(0.6, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.02, self.tr_size * 0.5)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Efficient GP update schedule
            if len(self.X_hist) >= n_init:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)