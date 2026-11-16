import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X, y = [], []
        # Initial sampling: Sobol-like or simple random, keep minimal to save budget for BO
        init_size = min(self.budget // 4, 2 * self.dim + 2)
        for _ in range(init_size):
            x = np.random.uniform(lb, ub)
            X.append(x)
            y.append(func(x))
            
        X, y = np.array(X), np.array(y)
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx], y[best_idx]
        
        def acquisition(x, gp, y_min):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            from scipy.stats import norm
            z = (y_min - mu) / sigma
            return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        for _ in range(init_size, self.budget):
            # Update GP with periodic hyperparameter tuning
            self.gp.fit(X, y)
            
            # Acquisition Optimization: Multi-start L-BFGS-B
            y_min = np.min(y)
            best_acq = np.inf
            next_x = None
            
            for _ in range(5):
                x0 = np.random.uniform(lb, ub)
                res = minimize(acquisition, x0, args=(self.gp, y_min), bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for acquisition
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x