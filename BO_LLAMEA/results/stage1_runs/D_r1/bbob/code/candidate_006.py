import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a flexible kernel with an adaptive noise floor
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using LHS-like approach for better space coverage
        n_init = max(2 * self.dim + 2, 10)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def neg_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, None)
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        for i in range(n_init, self.budget):
            # Periodically update hyperparameters; re-fitting every step for small budget is fine
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Acquisition optimization with multi-start
            best_acq = np.inf
            next_x = None
            
            # Candidates for L-BFGS-B: random points + best historical point
            candidates = np.random.uniform(lb, ub, (10, self.dim))
            candidates = np.vstack([candidates, best_x])
            
            for x0 in candidates:
                res = minimize(neg_ei, x0, args=(self.gp, y_min), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if acquisition fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            # Execute objective
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