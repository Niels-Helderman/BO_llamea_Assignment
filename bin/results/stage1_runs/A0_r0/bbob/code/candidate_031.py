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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def _get_lhs(self, n, lb, ub):
        samples = (np.argsort(np.random.rand(n, self.dim), axis=0) + np.random.rand(n, self.dim)) / n
        return lb + (ub - lb) * samples

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        n_init = min(self.budget, 2 * self.dim + 4)
        init_points = self._get_lhs(n_init, lb, ub)
        
        for x in init_points:
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to handle growing data
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
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
            
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Multi-start acquisition to avoid local minima
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start in starts:
                res = minimize(acquisition, start, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
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