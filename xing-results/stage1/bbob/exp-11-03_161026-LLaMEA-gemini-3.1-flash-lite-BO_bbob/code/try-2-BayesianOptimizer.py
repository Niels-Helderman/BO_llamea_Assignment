import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial design: Use more points for 6D (12 points) to capture global structure
        n_init = max(2 * self.dim, 12)
        
        best_y_curr = np.inf
        best_x_curr = None

        for i in range(self.budget):
            if i < n_init:
                # Stratified random sampling for initial coverage
                x_next = np.random.uniform(lb, ub)
            else:
                # Refit hyperparameters every 5 steps for efficiency
                if i % 5 == 0:
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                best_y = np.min(y_obs)
                
                def neg_log_ei(x):
                    x = x.reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    # Use Log-EI to avoid underflow in tails
                    log_ei = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z) + np.log1p(sigma/diff * norm.pdf(z)/norm.cdf(z) + 1e-20)
                    return -log_ei

                # Multi-start acquisition optimization using L-BFGS-B
                x_next = None
                best_acq = np.inf
                for _ in range(8):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(neg_log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), best_x_curr