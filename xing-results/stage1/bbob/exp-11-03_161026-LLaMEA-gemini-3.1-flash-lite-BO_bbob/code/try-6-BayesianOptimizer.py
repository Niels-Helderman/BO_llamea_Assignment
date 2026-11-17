import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        n_init = min(self.budget, 2 * self.dim + 2)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sobol.random(n_init)
        
        best_y_curr, best_x_curr = float('inf'), None
        
        for i in range(self.budget):
            if i < n_init:
                x_next = lb + (ub - lb) * init_points[i]
            else:
                if i % 3 == 0:
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                best_y = np.min(y_obs)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.clip(sigma, 1e-12, None)
                    with np.errstate(divide='ignore'):
                        z = (best_y - mu) / sigma
                        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                x_next = None
                best_acq = float('inf')
                for _ in range(10):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x
                if x_next is None: x_next = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr, best_x_curr = y_next, x_next
        
        return float(best_y_curr), np.array(best_x_curr)