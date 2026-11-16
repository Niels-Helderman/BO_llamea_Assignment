import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matérn 5/2 kernel with fixed signal variance and a tiny noise floor for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial sampling to 2*dim + 4 to provide a better GP prior in early stages
        n_init = max(2 * self.dim + 4, 12)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def neg_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            # Log-EI transformation is numerically more stable for optimization
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (y_min - mu) / sigma
                log_ei = np.log(y_min - mu) + norm.logcdf(z) + np.log1p(sigma / (y_min - mu) * norm.pdf(z) / norm.cdf(z) + 1e-12)
            return -np.nan_to_num(log_ei, nan=0.0)

        for i in range(n_init, self.budget):
            try:
                # Refit GP every iteration as budget is small; use warm-start logic by design of sklearn
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Acquisition optimization using multi-start L-BFGS-B
            best_acq = np.inf
            next_x = None
            
            # Use 15 candidates: random uniform for exploration, best known for exploitation
            candidates = np.random.uniform(lb, ub, (14, self.dim))
            candidates = np.vstack([candidates, best_x])
            
            for x0 in candidates:
                res = minimize(neg_ei, x0, args=(self.gp, y_min), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback to random if acquisition optimization fails or is singular
            if next_x is None or not np.all(np.isfinite(next_x)):
                next_x = np.random.uniform(lb, ub)
                
            val = func(next_x)
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x