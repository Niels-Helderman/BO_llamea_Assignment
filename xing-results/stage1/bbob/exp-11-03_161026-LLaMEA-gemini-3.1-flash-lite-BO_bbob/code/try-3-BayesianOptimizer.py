import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Mat¨¦rn 5/2 for smoothness and WhiteKernel for numerical jitter
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e1), nu=2.5) + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        
        # 1. Initial Design using random sampling to ensure coverage
        n_init = max(2 * self.dim, 5)
        
        best_y_curr = np.inf
        best_x_curr = None

        for i in range(self.budget):
            if i < n_init:
                x_next = np.random.uniform(lb, ub)
            else:
                # 2. Refit hyperparameters
                try:
                    self.gp.fit(X_obs, y_obs)
                except:
                    pass
                
                # 3. Acquisition function: Expected Improvement
                best_y = np.min(y_obs)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                # 4. Multi-start acquisition optimization
                x_next = np.random.uniform(lb, ub)
                best_acq = np.inf
                for _ in range(8):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), np.array(best_x_curr)