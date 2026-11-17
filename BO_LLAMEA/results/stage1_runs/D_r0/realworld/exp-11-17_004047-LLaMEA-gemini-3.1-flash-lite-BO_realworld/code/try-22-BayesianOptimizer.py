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
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
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
                # Latin Hypercube-style initialization
                x = lb + (ub - lb) * (np.random.rand(self.dim))
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                # Multistart acquisition optimization
                best_ei = np.inf
                best_x_norm = x_inc_norm
                for _ in range(5):
                    start_point = np.clip(x_inc_norm + np.random.normal(0, self.tr_size, self.dim), 0, 1)
                    res = minimize(
                        lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                        x0=start_point, bounds=[(0, 1)] * self.dim, method='L-BFGS-B'
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x_norm = res.x
                
                x = best_x_norm * (ub - lb) + lb
                self.tr_size = np.clip(self.tr_size * (1.1 if best_ei < -1e-6 else 0.7), 0.05, 0.8)
            
            x = np.clip(x, lb, ub)
            
            # Avoid duplicate queries
            if len(self.X_hist) > 0 and np.any(np.all(np.abs(self.X_hist - x) < 1e-6, axis=1)):
                x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            if len(self.X_hist) >= n_init:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)