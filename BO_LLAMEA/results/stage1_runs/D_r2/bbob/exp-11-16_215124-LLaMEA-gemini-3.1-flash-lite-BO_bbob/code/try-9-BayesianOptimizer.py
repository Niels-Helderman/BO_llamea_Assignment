import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel setup for black-box optimization
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial space-filling design using quasi-random sampling
        init_size = min(max(2 * self.dim + 2, 12), self.budget // 4)
        X, y = [], []
        for _ in range(init_size):
            x = np.random.uniform(lb, ub)
            X.append(x)
            y.append(func(x))
        
        X, y = np.array(X), np.array(y)
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        # Acquisition function: Expected Improvement
        def get_ei(x_norm):
            x = lb + x_norm * range_val
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            imp = best_y - mu
            Z = imp / sigma
            return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

        for i in range(init_size, self.budget):
            try:
                self.gp.fit(X, y)
                
                # Global acquisition optimization using L-BFGS-B multistart
                best_ei = -np.inf
                next_x = None
                
                # Generate candidate points for acquisition optimization
                starts = [np.random.uniform(0, 1, self.dim) for _ in range(8)]
                starts.append((best_x - lb) / range_val) # local exploit
                
                for x0 in starts:
                    res = minimize(get_ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x = lb + res.x * range_val
                
                if next_x is None: raise ValueError("Acquisition opt failed")
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
            
            # Periodically rebuild GP to prevent memory/scale issues if necessary
            if i % 100 == 0:
                self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
            
        return float(best_y), best_x