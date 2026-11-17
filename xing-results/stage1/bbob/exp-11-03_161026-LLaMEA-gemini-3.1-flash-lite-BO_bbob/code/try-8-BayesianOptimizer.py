import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        n_init = min(self.budget, 2 * self.dim + 2)
        
        # Sobol sequence for better space-filling initial coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = sampler.random(n=n_init)
        init_points = qmc.scale(init_samples, lb, ub)
        
        best_y_curr, best_x_curr = np.inf, None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                # Periodic refitting every few steps to save compute, but frequent enough for accuracy
                if i % 3 == 0 or i == n_init:
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                best_y = np.min(y_obs)
                
                def log_ei(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z = diff / sigma
                        log_ei_val = np.log(diff + 1e-15) + norm.logcdf(z) + np.log1p(sigma/diff * (norm.pdf(z) / norm.cdf(z) + 1e-15))
                    return -np.nan_to_num(log_ei_val, nan=-np.inf)

                # Multi-start acquisition optimization
                x_next = np.random.uniform(lb, ub)
                best_acq = np.inf
                for _ in range(10):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr, best_x_curr = y_next, x_next
        
        return float(best_y_curr), np.array(best_x_curr)