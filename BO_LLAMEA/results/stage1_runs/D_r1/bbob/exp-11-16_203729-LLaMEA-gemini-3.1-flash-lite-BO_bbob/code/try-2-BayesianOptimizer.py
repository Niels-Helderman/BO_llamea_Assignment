import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small white noise to handle deterministic objective jitter
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube-like sampling for better coverage than pure random
        init_size = max(2 * self.dim + 1, min(self.budget // 5, 10))
        X = np.array([np.random.uniform(lb, ub) for _ in range(init_size)])
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def log_ei(x, gp, y_min):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            diff = y_min - mu
            z = diff / sigma
            # Using Log-EI to prevent numerical underflow and emphasize tail exploration
            log_ei_val = diff * norm.cdf(z) + sigma * norm.pdf(z)
            return -np.log(np.maximum(log_ei_val, 1e-20))

        for i in range(init_size, self.budget):
            # Refit GP periodically or every step (cheap for N < 100)
            try:
                self.gp.fit(X, y)
            except:
                pass # Continue with previous GP state if fitting fails
            
            y_min = np.min(y)
            next_x = None
            best_acq = np.inf
            
            # Multi-start acquisition optimization
            for _ in range(8):
                x0 = np.random.uniform(lb, ub)
                res = minimize(log_ei, x0, args=(self.gp, y_min), bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            try:
                val = func(next_x)
            except:
                val = np.inf
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x