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
        # Robust kernel: Matern 5/2 is standard; WhiteKernel handles potential noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5 

    def _get_ei(self, x, gp, y_best):
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)
        
        # Initial phase: Latin Hypercube-like spread for coverage
        init_samples = min(self.budget, 2 * self.dim + 2)
        
        for i in range(self.budget):
            if i < init_samples:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                
                # Trust Region search: restrict optimization to local box
                tr_half = self.tr_size / 2.0
                curr_norm = (x_incumbent - lb) / (ub - lb)
                tr_lb = np.clip(curr_norm - tr_half, 0.0, 1.0)
                tr_ub = np.clip(curr_norm + tr_half, 0.0, 1.0)
                
                res = minimize(
                    lambda x_n: self._get_ei(x_n * (ub - lb) + lb, self.gp, y_best),
                    x0=np.clip(curr_norm, tr_lb, tr_ub),
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Update TR based on surrogate progress
                if res.fun < -1e-7:
                    self.tr_size = min(0.5, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.01, self.tr_size * 0.7)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                
            # Periodic GP refit
            if len(self.X_hist) >= 5:
                try:
                    X_arr = np.array(self.X_hist)
                    y_arr = np.array(self.y_hist)
                    self.gp.fit(X_arr, y_arr)
                except:
                    pass
                    
        return float(best_y), best_x