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
        # ARD Matern kernel is better for 6D real-world problems
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=1)
        self.tr_size = 0.5 

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = x_norm.reshape(1, -1) * (ub - lb) + lb
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        diff = y_best - mu
        z = diff / sigma
        return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)
        
        # Initial sampling: Use a centered approach to avoid extreme boundary bias
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # Deterministic jittered grid for better coverage than random
                x = lb + (ub - lb) * ((np.linspace(0, 1, n_init) + 0.1 * i) % 1.0)
                x = x if self.dim == 1 else lb + (ub - lb) * np.random.rand(self.dim)
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                # Acquisition optimization
                best_acq = np.inf
                x_next_norm = x_inc_norm
                
                for _ in range(5):
                    x0 = np.random.uniform(tr_lb, tr_ub)
                    res = minimize(lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub), 
                                   x0=x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next_norm = res.x
                
                x = x_next_norm * (ub - lb) + lb
                
                # Update trust region based on success
                if best_acq < -1e-6:
                    self.tr_size = min(0.5, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.01, self.tr_size * 0.5)

            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Refit GP
            if len(self.X_hist) >= n_init:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    self.gp.kernel = self.kernel
                    
        return float(best_y), np.array(best_x)