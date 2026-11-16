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
        # Increased alpha slightly for better numerical stability with deterministic functions
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-9, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube-like initial spread to cover 6D space effectively
        n_init = min(max(2 * self.dim, 10), self.budget // 4)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) if i > 0 else 0.5)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Efficient hyperparameter tuning schedule
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                pass
                
            y_min = np.min(y)
            
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            # Robust acquisition optimization using multi-start L-BFGS-B
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Use current best and random samples to seed acquisition optimization
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start_point in candidates:
                res = minimize(acquisition, start_point, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x