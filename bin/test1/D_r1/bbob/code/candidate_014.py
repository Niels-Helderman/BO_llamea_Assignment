import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Increased robustness: use a slightly higher noise_level for numerical stability
        # and optimize kernel hyperparameters only every few iterations to save cost.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-7)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling: Use a larger initial set for 6D to get a better prior surrogate
        n_init = min(self.budget // 4, 15)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def neg_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, None)
            diff = y_min - mu
            z = diff / sigma
            return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

        for i in range(n_init, self.budget):
            try:
                # Refit only when data is sufficient to avoid over-fitting early
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            best_acq = np.inf
            next_x = None
            
            # Efficient acquisition optimization: multi-start with random search for better exploration
            candidates = np.random.uniform(lb, ub, (15, self.dim))
            # Include local search starting from current best to improve exploitation
            candidates = np.vstack([candidates, best_x + np.random.normal(0, 0.05 * (ub - lb), (5, self.dim))])
            
            for x0 in candidates:
                x0 = np.clip(x0, lb, ub)
                res = minimize(neg_ei, x0, args=(self.gp, y_min), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback to random if acquisition optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            else:
                next_x = np.clip(next_x, lb, ub)
                
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