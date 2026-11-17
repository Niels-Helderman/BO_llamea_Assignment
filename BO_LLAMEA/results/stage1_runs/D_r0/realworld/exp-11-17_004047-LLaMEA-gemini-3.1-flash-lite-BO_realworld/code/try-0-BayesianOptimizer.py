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
        
        # Kernel: Matern 5/2 is standard for black-box optimization
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2)

    def _get_ei(self, x, gp, y_best):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei  # Minimize negative EI

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            # Initial sampling (LHS-like)
            if i < 2 * self.dim + 2:
                x = np.random.uniform(lb, ub)
            else:
                # Acquisition optimization
                y_best = np.min(self.y_hist)
                x_start = self.X_hist[np.argmin(self.y_hist)]
                
                res = minimize(
                    lambda x_norm: self._get_ei(x_norm * (ub - lb) + lb, self.gp, y_best),
                    x0=(x_start - lb) / (ub - lb),
                    bounds=[(0, 1)] * self.dim,
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
            
            # Ensure within bounds due to float precision
            x = np.clip(x, lb, ub)
            
            # Objective call
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Update GP surrogate
            if len(self.X_hist) >= 3:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass # Keep previous model if fit fails
                    
        return float(best_y), best_x