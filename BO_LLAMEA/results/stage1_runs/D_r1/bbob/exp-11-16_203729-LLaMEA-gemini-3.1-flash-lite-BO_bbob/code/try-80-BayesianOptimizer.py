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
        # Use alpha for jitter to handle deterministic function interpolation
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True)
        self.tr_length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial points
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                # Fallback to random if GP fitting fails
                next_x = np.random.rand(self.dim)
            else:
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    z = (best_y - mu) / sigma
                    return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))

                # Localized trust region + random exploration
                best_acq = np.inf
                next_x = None
                
                # Propose points from local trust region and global uniform space
                candidates = []
                for _ in range(10):
                    candidates.append(np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1))
                for _ in range(5):
                    candidates.append(np.random.rand(self.dim))
                    
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            
            # Ensure next_x is valid
            next_x = np.clip(next_x, 0, 1)
            val = func(next_x * (ub - lb) + lb)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Trust region update logic
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_length = min(self.tr_length * 1.1, 1.0)
            else:
                self.tr_length = max(self.tr_length * 0.7, 0.05)
                # Restart TR if it shrinks too much
                if self.tr_length < 0.06:
                    self.tr_length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb