import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        # Initial sampling: Sobol-like simple quasi-random initialization
        n_init = min(12, self.budget // 4)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim))
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]

        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / sigma
                return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit hyperparameters
            if i % 10 == 0:
                self.gp.kernel = self.kernel
                self.gp.fit(X_data, y_data)
            else:
                self.gp.fit(X_data, y_data)
            
            # Maximize EI using L-BFGS-B with multi-start
            y_min = np.min(y_data)
            best_acq = -np.inf
            next_x = None
            
            for _ in range(5):
                x0 = lb + (ub - lb) * np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x