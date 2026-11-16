import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a more flexible kernel configuration to better capture BBOB landscape
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Latin Hypercube-like spacing
        init_size = max(2 * self.dim + 1, min(self.budget // 4, 15))
        X = np.array([np.random.uniform(lb, ub) for _ in range(init_size)])
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def acquisition(x, gp, y_min):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            # Standard EI: maximizing improvement
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Minimize negative EI

        for i in range(init_size, self.budget):
            # Refit GP with regularization
            try:
                self.gp.fit(X, y)
            except:
                pass 
            
            y_min = np.min(y)
            next_x = None
            best_acq = np.inf
            
            # Multi-start acquisition optimization using L-BFGS-B
            # Use Sobol or random points for better coverage
            for _ in range(12):
                x0 = np.random.uniform(lb, ub)
                res = minimize(acquisition, x0, args=(self.gp, y_min), bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
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