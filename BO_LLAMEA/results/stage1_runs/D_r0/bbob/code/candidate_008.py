import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Increased initial samples (n_init = 2 * dim + 5) to jump-start the model
        n_init = 2 * self.dim + 5
        X_obs, y_obs = [], []
        
        # Structured LHS-style initialization
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) + np.arange(self.dim)) / (n_init + self.dim)
            x = np.clip(x, lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_scale = 0.5
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Fit GP with warm-start logic
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            next_x = None
            best_acq = -1.0
            
            # Hybrid search: Global + Trust Region
            search_attempts = [best_x + tr_scale * range_val * (np.random.rand(self.dim) - 0.5),
                               lb + (ub - lb) * np.random.rand(self.dim)]
            
            for x0 in search_attempts:
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
                best_y, best_x = float(y_new), next_x
                tr_scale = min(0.6, tr_scale * 1.2)
            else:
                tr_scale = max(0.1, tr_scale * 0.8)
                
        return best_y, best_x