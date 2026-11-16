import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design using LHS for better distribution in 6D
        n_init = 2 * self.dim + 2
        sampler = qmc.LatinHypercube(d=self.dim)
        X_init = qmc.scale(sampler.random(n=n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        def get_ei(x, gp, y_min):
            x = np.atleast_2d(x)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            y_min = np.min(y_data)
            
            # Periodically re-optimize hyperparameters to save compute
            try:
                if i % 10 == 0:
                    self.gp.optimizer = 'fmin_l_bfgs_b'
                else:
                    self.gp.optimizer = []
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Acquisition optimization using multi-start L-BFGS-B
            next_x = None
            best_acq = -np.inf
            
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(8)]
            for x0 in candidates:
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
                
        return best_y, best_x