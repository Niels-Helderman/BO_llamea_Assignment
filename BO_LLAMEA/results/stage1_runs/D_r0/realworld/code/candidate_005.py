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
        # Matern 5/2 is robust for real-world functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5 

    def _get_ei(self, x, gp, y_best):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        # Expected Improvement calculation
        imp = y_best - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        # Initial exploration: use a dense enough start to populate the GP
        init_size = min(self.budget, 2 * self.dim + 4)
        
        for i in range(self.budget):
            if i < init_size:
                # Latin-hypercube-lite sampling for better coverage
                x = lb + (ub - lb) * (np.random.rand(self.dim) + (i % 2)) / 2.0
                x = np.clip(x, lb, ub)
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                
                # Normalize incumbent to [0, 1]
                x_norm = (x_incumbent - lb) / (ub - lb)
                
                # Trust Region adaptation
                tr_l = np.clip(x_norm - self.tr_size, 0, 1)
                tr_u = np.clip(x_norm + self.tr_size, 0, 1)
                
                # Acquisition optimization
                res = minimize(
                    lambda x_n: self._get_ei(x_n * (ub - lb) + lb, self.gp, y_best),
                    x0=x_norm,
                    bounds=list(zip(tr_l, tr_u)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Update trust region based on success
                if res.fun < -1e-6:
                    self.tr_size = min(0.5, self.tr_size * 1.2)
                else:
                    self.tr_size = max(0.05, self.tr_size * 0.7)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Incremental fitting
            if len(self.X_hist) >= 3:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), best_x