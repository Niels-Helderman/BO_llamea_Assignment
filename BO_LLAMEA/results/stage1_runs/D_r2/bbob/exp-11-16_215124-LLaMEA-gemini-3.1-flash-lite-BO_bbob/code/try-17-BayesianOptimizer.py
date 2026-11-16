import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel for better feature scaling adaptability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Latin Hypercube-like initialization for better coverage
        init_size = 2 * self.dim + 4
        X, y = [], []
        for i in range(init_size):
            x = lb + (np.random.rand(self.dim) + i) / (init_size + 1) * range_val
            X.append(x)
            y.append(func(x))
        
        X, y = np.array(X), np.array(y)
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        for i in range(init_size, self.budget):
            # Periodically rebuild to manage memory and computational cost
            if i % 15 == 0:
                self.gp.kernel_ = self.kernel
                self.gp.fit(X, y)
            else:
                self.gp.fit(X, y)
            
            def objective(x_norm):
                x = lb + x_norm * range_val
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.clip(sigma, 1e-12, None)
                diff = best_y - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Optimize acquisition with multi-start
            best_ei = -1.0
            next_x = None
            for _ in range(8):
                x0 = np.random.rand(self.dim)
                res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    next_x = lb + res.x * range_val
            
            if next_x is None:
                next_x = lb + np.random.rand(self.dim) * range_val
            
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x