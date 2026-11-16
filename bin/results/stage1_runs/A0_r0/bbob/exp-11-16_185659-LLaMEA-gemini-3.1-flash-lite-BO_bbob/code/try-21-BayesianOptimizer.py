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
        
        # Initial sampling using Latin Hypercube-style random seeding
        n_init = min(2 * self.dim + 2, self.budget // 4)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Efficient hyperparameter tuning frequency
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            def ei_acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Multi-start optimization of the acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            
            # Candidates include current best, random points, and local search
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start in starts:
                res = minimize(ei_acquisition, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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