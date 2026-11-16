import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Using a fixed alpha for numerical stability; avoid excessive hyperparameter re-fitting
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True)
        self.trust_region_size = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol sequence
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Refit every step, but with warm start on kernel hyperparameters
                self.gp.fit(X, y)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                # Adaptive search: Local (around best) + Global (Sobol/random)
                best_acq = np.inf
                next_x = None
                
                # Trust-region candidates
                lb_tr = np.clip(best_x - self.trust_region_size, 0, 1)
                ub_tr = np.clip(best_x + self.trust_region_size, 0, 1)
                
                candidates = [np.random.uniform(lb_tr, ub_tr) for _ in range(10)]
                candidates.append(np.random.rand(self.dim)) # Add one global exploration point
                
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            except:
                next_x = np.random.rand(self.dim)
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.rand(self.dim)
                
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                self.trust_region_size = np.clip(self.trust_region_size * 1.1, 0.05, 1.0)
            else:
                self.trust_region_size = np.clip(self.trust_region_size * 0.9, 0.05, 1.0)
                # Restart TR if it shrinks too small
                if self.trust_region_size <= 0.06:
                    self.trust_region_size = 0.5
                    
        return best_y, best_x * (ub - lb) + lb