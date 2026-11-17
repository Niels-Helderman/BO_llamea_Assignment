import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        
        # Initial Design: Latin Hypercube-like spread using Sobol-sequence or random
        n_init = min(self.budget // 5, 2 * self.dim + 2)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = np.random.uniform(lb, ub)
            else:
                # Refit hyperparameters every few steps to save cost
                if i % 8 == 0:
                    self.gp.fit(X_obs, y_obs)
                
                # Acquisition function: Expected Improvement
                best_y = np.min(y_obs)
                
                def acq(x):
                    x = x.reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    from scipy.stats import norm
                    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                # Multi-start acquisition optimization
                x_next = None
                best_acq = np.inf
                for _ in range(5):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if i == 0 or y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), best_x_curr