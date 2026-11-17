import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a flexible kernel with a small constant white noise to handle deterministic data
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling: Increase to 2*dim + 4 for better coverage in 6D
        n_init = 2 * self.dim + 4
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Acquisition function: Log-EI for improved numerical stability
        def get_log_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, None)
            diff = y_min - mu
            z = diff / sigma
            # Log-EI: log(diff * Phi(z) + sigma * phi(z))
            return np.log(np.maximum(diff * norm.cdf(z) + sigma * norm.pdf(z), 1e-30))

        # Refit GP every few iterations to balance cost and accuracy
        for i in range(n_init, self.budget):
            try:
                if i % 2 == 0:
                    self.gp.fit(X, y)
                
                y_min = np.min(y)
                
                # Multi-start acquisition optimization with global exploration
                best_acq = -np.inf
                next_x = None
                
                # Combine random search with local optimization (L-BFGS-B)
                for _ in range(12):
                    x0 = np.random.uniform(0, 1, self.dim)
                    res = minimize(lambda x: -get_log_ei(x, self.gp, y_min), x0, 
                                   bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if -res.fun > best_acq:
                        best_acq = -res.fun
                        next_x = res.x
            except:
                # Fallback to pure exploration if GP fitting fails
                next_x = np.random.uniform(0, 1, self.dim)
            
            if next_x is None: 
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
            
        return best_y, best_x_norm * (ub - lb) + lb