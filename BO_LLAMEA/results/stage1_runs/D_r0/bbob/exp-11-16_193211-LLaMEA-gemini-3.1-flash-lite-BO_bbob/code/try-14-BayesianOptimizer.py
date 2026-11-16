import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Constant scale, Matern kernel with ARD (nu=2.5 for smoothness)
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, 10)
        
        # Initialization with Sobol sequence for better space filling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        def log_ei(x, gp, y_min, jitter):
            x = np.atleast_2d(x)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            # Log-EI is more stable for optimization
            diff = y_min - mu - jitter
            z = diff / sigma
            log_val = np.log(sigma) + norm.logcdf(z) + np.log(norm.pdf(z) / norm.cdf(z) + z)
            return -np.exp(log_val)

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            # Adaptive jitter: reduce exploration as we get closer to the budget limit
            jitter = 0.01 * (1.0 - (i / self.budget))
            
            # Multi-start acquisition optimization using L-BFGS-B
            best_acq = np.inf
            next_x = None
            
            # Use Latin Hypercube or simple uniform random for acquisition restarts
            starts = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(5)]
            for x0 in starts:
                res = minimize(lambda x: log_ei(x, self.gp, y_min, jitter), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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