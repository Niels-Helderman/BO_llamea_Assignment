import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        n_init = max(2 * self.dim + 2, 10)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) if i > 0 else 0.5)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_y, best_x = y_obs[best_idx], X_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X, y = np.array(X_obs), np.array(y_obs)
            # Retrain hyperparameters every 8 steps to balance speed and model quality
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            self.gp.fit(X, y)
            
            y_min = np.min(y)
            # Use Expected Improvement with a small exploration penalty/jitter for numerical stability
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Multi-start optimization with local search around current best and random candidates
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for start in starts:
                res = minimize(acquisition, start, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-6)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                
        return float(best_y), best_x