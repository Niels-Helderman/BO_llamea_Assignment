import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel for automatic relevance determination
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim, 12)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            # Refit kernel hyperparameters every 20 steps to save cost
            if i % 20 == 0 or i == n_init:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                pass
            
            # Expected Improvement acquisition
            def ei(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            # Multi-start gradient-based optimization of EI
            next_x = None
            best_acq = np.inf
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            
            for x0 in starts:
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            y_new = func(next_x)
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x