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
        # Use a small alpha for numerical stability; normalize_y helps handle varying scales
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial points: latin hypercube style seeding
        n_init = min(2 * self.dim + 2, self.budget // 4)
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
            
            # Optimize GP hyperparameters every 8 steps to balance cost and accuracy
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Acquisition optimization with multiple restarts
            best_acq = np.inf
            candidate = np.random.uniform(lb, ub)
            
            seeds = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for s in seeds:
                res = minimize(ei, s, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    candidate = res.x
            
            y_new = func(candidate)
            X_obs.append(candidate)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = candidate
                
        return float(best_y), best_x