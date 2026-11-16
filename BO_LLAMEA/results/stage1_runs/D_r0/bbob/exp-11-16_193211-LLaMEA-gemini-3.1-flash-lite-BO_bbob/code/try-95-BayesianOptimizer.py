import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel for automatic relevance determination of dimensions
        # Add WhiteKernel to handle potential numerical jitter in deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0,  # Alpha handled by WhiteKernel
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            # Refit GP every few steps or when data grows to manage compute
            if (i - n_init) % 5 == 0:
                self.gp.n_restarts_optimizer = 1
                try:
                    self.gp.fit(X_obs, y_obs)
                except:
                    pass
            
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # Use EI acquisition rule
                with np.errstate(divide='ignore'):
                    z = (best_y - mu) / sigma
                    ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Global optimization for next candidate using L-BFGS-B
            best_acq = -np.inf
            next_x = None
            
            # Multi-start acquisition optimization
            for _ in range(5):
                start_x = np.random.uniform(lb, ub)
                res = minimize(acquisition, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if acquisition optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            
            # Update best
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
            
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
                
        return best_y, best_x