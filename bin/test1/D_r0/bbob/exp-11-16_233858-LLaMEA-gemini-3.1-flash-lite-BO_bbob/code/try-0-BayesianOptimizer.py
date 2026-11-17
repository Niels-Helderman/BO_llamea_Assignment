import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        # Initial sampling using a small Latin Hypercube-like approach
        n_init = min(12 + 2 * self.dim, self.budget // 4)
        for i in range(n_init):
            x = np.random.uniform(lb, ub)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        best_y, best_x = y_obs[best_idx], X_obs[best_idx]
        
        # Optimization loop
        while len(X_obs) < self.budget:
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-fit hyperparameters
            if len(X_obs) % 10 == 0:
                self.gp.kernel = self.kernel
                self.gp.fit(X_data, y_data)
            else:
                self.gp.fit(X_data, y_data)
            
            # Acquisition function: Expected Improvement
            y_min = np.min(y_data)
            
            def ei(x):
                x = x.reshape(1, -1)
                mu, std = self.gp.predict(x, return_std=True)
                with np.errstate(divide='ignore'):
                    z = (y_min - mu) / (std + 1e-9)
                    ei_val = (y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
                return -ei_val[0]

            # Multi-start acquisition optimization
            best_acq = -np.inf
            next_x = X_data[np.argmin(y_data)]
            
            for _ in range(5):
                start_x = np.random.uniform(lb, ub)
                res = minimize(ei, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Query function
            y_next = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_next)
            
            if y_next < best_y:
                best_y, best_x = y_next, next_x
                
        return float(best_y), best_x