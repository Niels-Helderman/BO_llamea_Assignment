import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use WhiteKernel for numerical stability on deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_dim = ub - lb
        
        n_init = max(2 * self.dim + 1, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            # Fit GP (refit hyperparameters every few steps to save compute)
            if i % 10 == 0 or i == n_init:
                try:
                    self.gp.fit(X_obs, y_obs)
                except:
                    pass
            
            # Log-Expected Improvement Acquisition
            def log_ei(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI is more stable than EI for small improvements
                val = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -val if val > 1e-12 else 0.0

            # Optimize acquisition with multi-start
            best_acq = np.inf
            next_x = None
            
            # Start points: incumbent + random candidates
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for x0 in starts:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for numerical failure or zero acquisition
            if next_x is None or best_acq == 0.0:
                next_x = np.random.uniform(lb, ub)
            
            # Ensure next_x is distinct from existing to avoid GP singularity
            dist = np.linalg.norm(X_obs - next_x, axis=1)
            if np.min(dist) < 1e-6:
                next_x = np.random.uniform(lb, ub)

            y_new = func(next_x)
            
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x.copy()
        
        return best_y, best_x