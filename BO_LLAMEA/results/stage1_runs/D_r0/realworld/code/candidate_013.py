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
        # Robust kernel: Matern 5/2 is standard for BBO, WhiteKernel handles noise
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        self.tr_size = 0.5 

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-12, None)
        with np.errstate(divide='ignore'):
            imp = y_best - mu
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        # Initial sampling: 2*dim + 2 points ensures sufficient variance for GP
        n_init = max(2 * self.dim + 2, 12)
        
        for i in range(self.budget):
            if i < n_init:
                # Latin Hypercube Sampling logic
                x = lb + (ub - lb) * (np.random.rand(self.dim) + (i % n_init)) / n_init
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                # Trust Region search for local exploitation
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                # Multiple starts for acquisition optimizer
                best_ei = np.inf
                x_best_ei = x_inc_norm
                for _ in range(3):
                    x0 = np.random.uniform(tr_lb, tr_ub)
                    res = minimize(self._get_ei, x0=x0, args=(self.gp, y_best, lb, ub),
                                   bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if res.fun < best_ei:
                        best_ei = res.fun
                        x_best_ei = res.x
                
                x = x_best_ei * (ub - lb) + lb
                
                # Adjust Trust Region
                if best_ei < -1e-6:
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
            
            # Re-fit GP every call after initial batch, limited by budget constraints
            if len(self.X_hist) >= 5:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)