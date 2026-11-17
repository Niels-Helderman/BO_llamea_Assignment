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
        # Use a kernel that balances smoothness and noise handling
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5 

    def _get_ei(self, x, gp, y_best):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        # Initial sampling: Use a Sobol-like spread or more dense LHS if budget allows
        init_size = min(self.budget, 2 * self.dim + 2)
        
        for i in range(self.budget):
            if i < init_size:
                # Latin Hypercube-style initial sampling
                x = lb + (ub - lb) * (np.random.rand(self.dim) + np.arange(self.dim)) / (self.dim + 1)
                x = np.clip(x, lb, ub)
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                
                # Normalize incumbent for TR logic
                x_norm = (x_incumbent - lb) / (ub - lb)
                tr_lb = np.maximum(0.0, x_norm - self.tr_size)
                tr_ub = np.minimum(1.0, x_norm + self.tr_size)
                
                # Optimize acquisition
                res = minimize(
                    lambda x_n: self._get_ei(x_n * (ub - lb) + lb, self.gp, y_best),
                    x0=x_norm,
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Adaptive TR scaling
                if res.fun < -1e-7: 
                    self.tr_size = min(0.5, self.tr_size * 1.2)
                else:
                    self.tr_size = max(0.05, self.tr_size * 0.5)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Fit GP only if enough data points
            if len(self.X_hist) >= 3:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), best_x