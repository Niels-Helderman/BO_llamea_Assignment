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
        # alpha provides numerical stability for deterministic objective functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial design using latin hypercube-like sampling for better coverage
        n_init = min(15, self.budget // 4)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) if i > 0 else 0.5)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Efficient hyperparameter fitting schedule
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
                return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Acquisition optimization with multi-start
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Mix local search from current best and global random samples
            seeds = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for s in seeds:
                res = minimize(acquisition, s, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
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