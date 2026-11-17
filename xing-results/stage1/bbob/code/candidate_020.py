import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc
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
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = qmc.scale(sobol.random(n_init), lb, ub)
        
        best_y_curr = np.inf
        best_x_curr = None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_samples[i]
            else:
                if i % 3 == 0:
                    self.gp.fit(X_obs, y_obs)
                
                best_y = np.min(y_obs)
                def log_ei(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    # log(EI) = log(diff * Phi(z) + sigma * phi(z))
                    # for stability, use log-sum-exp trick or direct EI for small ranges
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                x_next = lb + (ub - lb) * np.random.rand(self.dim)
                best_val = np.inf
                for _ in range(10):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-5)
                    if res.fun < best_val:
                        best_val = res.fun
                        x_next = res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), np.array(best_x_curr)