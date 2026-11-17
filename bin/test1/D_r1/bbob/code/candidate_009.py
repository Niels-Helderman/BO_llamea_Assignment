import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small white noise for numerical stability in deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using small-scale LHS
        n_init = max(2 * self.dim + 2, 8)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def neg_ei(x, y_min):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            sigma = np.clip(sigma, 1e-15, None)
            diff = y_min - mu
            z = diff / sigma
            ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        for i in range(n_init, self.budget):
            # Refit GP every iteration (cheap for small budgets)
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Global acquisition optimization using DE for better exploration than L-BFGS-B
            bounds = list(zip(lb, ub))
            res = differential_evolution(neg_ei, bounds, args=(y_min,), 
                                         strategy='best1bin', popsize=10, tol=0.01, mutation=(0.5, 1), recombination=0.7)
            
            next_x = res.x if res.success or i % 3 != 0 else np.random.uniform(lb, ub)
            
            # Execute objective with fallback
            try:
                val = func(next_x)
            except:
                val = np.max(y) # Penalty
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x