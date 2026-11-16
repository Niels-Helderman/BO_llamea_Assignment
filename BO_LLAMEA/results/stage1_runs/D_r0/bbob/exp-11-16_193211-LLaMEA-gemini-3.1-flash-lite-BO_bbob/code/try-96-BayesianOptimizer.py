import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD (Automatic Relevance Determination) by setting length_scale to be a vector
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0.0, # Handled by WhiteKernel
            normalize_y=True,
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial exploration
        n_init = 2 * self.dim + 5
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = float(y_obs[best_idx])
        
        # Iterative optimization
        for i in range(n_init, self.budget):
            # Periodically re-optimize hyperparameters to handle changing landscape
            self.gp.n_restarts_optimizer = 5 if (i % 50 == 0) else 0
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                # Fallback to current best if fit fails
                next_x = best_x + np.random.normal(0, 0.01, self.dim) * (ub - lb)
            else:
                def ei(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    z = (best_y - mu) / sigma
                    return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start optimization of EI to avoid getting stuck in local optima
                starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(3)]
                best_ei = float('inf')
                next_x = best_x
                for start in starts:
                    res = minimize(ei, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_ei:
                        best_ei = res.fun
                        next_x = res.x
            
            # Ensure boundaries
            next_x = np.clip(next_x, lb, ub)
            
            # Prevent duplicate sampling
            if np.any(np.all(np.isclose(X_obs, next_x), axis=1)):
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x