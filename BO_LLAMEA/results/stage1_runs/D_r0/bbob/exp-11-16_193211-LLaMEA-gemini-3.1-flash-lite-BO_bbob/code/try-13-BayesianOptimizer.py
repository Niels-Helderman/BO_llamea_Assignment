import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel for individual length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 4
        
        # Latin Hypercube Sampling for improved space-filling initialization
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        def ei(x, gp, y_best):
            x = np.atleast_2d(x)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            improvement = y_best - mu
            z = improvement / sigma
            return -(improvement * norm.cdf(z) + sigma * norm.pdf(z))

        for i in range(n_init, self.budget):
            # Target log-transform stabilizes variance in non-stationary BBOB functions
            y_data = np.array(y_obs)
            y_log = np.log(y_data - np.min(y_data) + 1e-6)
            
            try:
                self.gp.fit(np.array(X_obs), y_log)
            except:
                pass
            
            # Multi-start acquisition optimization using L-BFGS-B
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(lambda x: ei(x, self.gp, np.min(y_log)), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x