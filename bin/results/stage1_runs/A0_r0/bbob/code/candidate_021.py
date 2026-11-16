import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        # Use a small constant alpha for numerical stability in deterministic scenarios
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube-like initialization for better coverage
        n_init = max(2 * self.dim + 2, 10)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim))
            X_obs.append(x)
            y_obs.append(func(x))
        
        best_idx = np.argmin(y_obs)
        best_y = y_obs[best_idx]
        best_x = X_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to balance cost/performance
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            def objective(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]
            
            # Multi-start optimization of the acquisition function
            next_x = best_x
            best_acq = np.inf
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(5)]
            
            for start_x in candidates:
                res = minimize(objective, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-6)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Ensure exploration if acquisition optimization gets stuck
            if np.linalg.norm(next_x - best_x) < 1e-5:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
                
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x