import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use ARD Mat¨¦rn kernel for better handling of individual dimensions
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=2
        )

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial points using Sobol sequence
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        best_y_curr = float('inf')
        best_x_curr = None
        
        for i in range(self.budget):
            if i < n_init:
                x_next = lb + (ub - lb) * init_points[i]
            else:
                # Refit GP
                self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                # Acquisition function: Log-EI for numerical stability
                y_min = np.min(y_obs)
                
                def neg_ei(x):
                    mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                    std = np.maximum(std, 1e-12)
                    diff = y_min - mu
                    z = diff / std
                    # Log-EI is more stable; maximize EI is equivalent to maximizing log(EI)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ei = diff * norm.cdf(z) + std * norm.pdf(z)
                    return -ei if ei > 0 else 0

                # Multi-start acquisition optimization
                x_next = lb + (ub - lb) * np.random.rand(self.dim)
                best_val = float('inf')
                for _ in range(8):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(neg_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val, x_next = res.fun, res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr, best_x_curr = y_next, x_next
        
        return best_y_curr, np.array(best_x_curr)