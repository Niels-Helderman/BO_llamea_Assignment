import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Using ARD kernel (length_scale is array of shape (dim,))
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 2
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            # Target transformation: log(y - min_y + eps) to handle scale and positivity
            y_curr = np.array(y_obs)
            y_min = np.min(y_curr)
            y_trans = np.log(y_curr - y_min + 1e-6)
            
            # Fit GP on transformed data
            X_data = np.array(X_obs)
            try:
                self.gp.fit(X_data, y_trans)
            except:
                pass
            
            # Acquisition: Expected Improvement on log-transformed surface
            def ei(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                target = np.min(y_trans)
                z = (target - mu) / sigma
                return -(sigma * (z * norm.cdf(z) + norm.pdf(z)))

            # Optimization within Trust Region
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            best_acq = np.inf
            next_x = None
            # Multi-start from incumbent and random points
            seeds = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(5)]
            for x0 in seeds:
                res = minimize(ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2) # Expand on success
            else:
                tr_size = max(0.05, tr_size * 0.8) # Shrink on failure
                
        return best_y, best_x