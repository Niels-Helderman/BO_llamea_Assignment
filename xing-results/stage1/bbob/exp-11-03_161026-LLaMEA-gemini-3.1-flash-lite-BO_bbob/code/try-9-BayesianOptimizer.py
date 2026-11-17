import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        n_init = max(2 * self.dim + 1, 10)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_points = lb + (ub - lb) * sobol.random(n_init)
        
        best_y_curr, best_x_curr = float('inf'), None
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                if i % 2 == 0:
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                best_y = np.min(y_obs)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                x_next = None
                best_acq = 1.0
                
                # Multi-start optimization with candidates from a Latin Hypercube
                candidates = lb + (ub - lb) * qmc.LatinHypercube(d=self.dim).random(20)
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-5)
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x
                
                if x_next is None:
                    x_next = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr, best_x_curr = y_next, x_next
        
        return float(best_y_curr), np.array(best_x_curr)