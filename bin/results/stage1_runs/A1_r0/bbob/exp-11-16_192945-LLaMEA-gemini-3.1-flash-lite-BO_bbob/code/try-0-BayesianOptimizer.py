import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern kernel for flexibility; add WhiteKernel to handle numerical jitter
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        
        # Initialization: Use a small Latin Hypercube-like random sample
        n_init = min(10, self.budget // 5)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = np.random.uniform(lb, ub)
            else:
                # Refit hyperparameters every few steps to save compute
                if i % 5 == 0:
                    self.gp.kernel.set_params(**self.gp.kernel.get_params())
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                # Optimize Acquisition Function (Expected Improvement)
                best_y = np.min(y_obs)
                x_next = self._propose_next(best_y, lb, ub)
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), X_obs[best_idx]

    def _propose_next(self, best_y, lb, ub):
        def acq(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            improvement = best_y - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei # Minimize negative EI

        # Multi-start optimization of the acquisition function
        best_acq = np.inf
        best_x = None
        for _ in range(5):
            x0 = np.random.uniform(lb, ub)
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        return best_x