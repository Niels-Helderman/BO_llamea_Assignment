import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a more flexible kernel with explicit length scale bounds to prevent collapse
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-11, 1e-4))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs, y_obs = [], []
        # Increased initial design density for better global surrogate mapping
        n_init = max(2 * self.dim, min(self.budget // 3, 15))
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                X = np.array(X_obs)
                y = np.array(y_obs)
                # Refit hyperparameters every 8 steps to balance cost and accuracy
                self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
                try:
                    self.gp.fit(X, y)
                except:
                    pass
                
                y_min = np.min(y)
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    z = (y_min - mu) / sigma
                    return -(sigma * (z * norm.cdf(z) + norm.pdf(z)))

                # Multi-start acquisition optimization
                best_acq, next_x = np.inf, None
                for _ in range(10):
                    start_x = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(acq, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
                x = next_x
            
            y_val = func(x)
            X_obs.append(x)
            y_obs.append(y_val)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), X_obs[best_idx]