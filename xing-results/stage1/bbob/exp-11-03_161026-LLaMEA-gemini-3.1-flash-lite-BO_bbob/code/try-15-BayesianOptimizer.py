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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        n_init = max(5, 2 * self.dim)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                X_arr, y_arr = np.array(X_obs), np.array(y_obs)
                # Periodically re-fit hyperparameters, but alpha remains stable
                self.gp.fit(X_arr, y_arr)
                
                best_y_curr = np.min(y_arr)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y_curr - mu
                    z = diff / sigma
                    # Use Negative EI
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                best_acq = np.inf
                x_next = init_points[np.random.choice(n_init)]
                
                # Multi-start optimization for acquisition
                for _ in range(10):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y:
                best_y = y_next
                best_x = x_next
        
        return float(best_y), np.array(best_x)