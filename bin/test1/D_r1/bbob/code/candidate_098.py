import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use WhiteKernel for numerical stability with deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial points for better surrogate initialization
        n_init = min(self.budget // 2, 4 * self.dim)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                
                # Log-EI is often more robust to numerical precision issues
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start optimization of the acquisition function
                best_acq = np.inf
                next_x = None
                
                # Sample points for multi-start
                candidates = np.random.rand(16, self.dim)
                # Bias candidates toward current best
                candidates = np.vstack([candidates, np.clip(best_x_norm + np.random.normal(0, 0.2, (8, self.dim)), 0, 1)])
                
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            except Exception:
                # Fallback to random point if GP optimization or fitting fails
                next_x = np.random.rand(self.dim)
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb