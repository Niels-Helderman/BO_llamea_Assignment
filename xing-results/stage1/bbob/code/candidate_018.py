import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use anisotropic kernel to allow different length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        n_init = min(self.budget, 2 * self.dim + 4)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        
        best_y_val = np.inf
        best_x_val = None

        for i in range(self.budget):
            if i < n_init:
                x_next = lb + (ub - lb) * sobol.random(1)[0]
            else:
                # Refit GP periodically to balance compute vs accuracy
                if i % 2 == 0:
                    try:
                        self.gp.fit(np.array(X_obs), np.array(y_obs))
                    except:
                        pass
                
                y_min = np.min(y_obs)
                
                def neg_ei(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    z = (y_min - mu) / sigma
                    ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                # Multi-start optimization of EI
                best_acq = np.inf
                x_next = np.random.uniform(lb, ub)
                # Candidates from random sampling + local optimization
                for _ in range(8):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(neg_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_val:
                best_y_val = y_next
                best_x_val = x_next
        
        return float(best_y_val), np.array(best_x_val)