import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e2))
        # Use a slightly larger alpha for numerical stability in deterministic scenarios
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial sampling using LHS-like random distribution for better coverage
        n_init = max(2 * self.dim + 2, 14)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Efficient hyperparameter update schedule
            optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            self.gp.optimizer = optimizer
            self.gp.fit(X, y)
            
            y_min = best_y
            
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # Use Expected Improvement
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            # Multi-start acquisition optimization from diverse candidates
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for start_point in candidates:
                res = minimize(acquisition, start_point, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Clamp next_x to ensure it's within bounds
            next_x = np.clip(next_x, lb, ub)
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x