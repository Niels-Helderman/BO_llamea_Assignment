import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel for individual length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_fixed=False) + WhiteKernel(noise_level=1e-8, noise_level_fixed=False)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial points using Sobol
        init_size = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        # Log-transformed target for better surrogate stability
        y_log = np.log(y - np.min(y) + 1e-6)
        
        for i in range(init_size, self.budget):
            try:
                # Refit GP every 5 steps to save time, else use existing hyperparameters
                if i % 5 == 0:
                    self.gp.fit(X, y_log)
                
                def get_ei(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    # Expected Improvement
                    curr_best_log = np.min(y_log)
                    Z = (curr_best_log - mu) / sigma
                    return -(sigma * (Z * norm.cdf(Z) + norm.pdf(Z)))

                # Multi-start optimization of EI
                best_ei = -np.inf
                next_x_norm = None
                for _ in range(6):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            y_log = np.log(y - np.min(y) + 1e-6)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x