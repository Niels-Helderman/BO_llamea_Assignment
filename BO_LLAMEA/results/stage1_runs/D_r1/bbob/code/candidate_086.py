import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel setup with explicit noise management
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Dense initial sampling for better global coverage at start
        n_init = min(self.budget // 3, 20)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    # Expected Improvement implementation
                    diff = best_y - mu
                    z = diff / sigma
                    return -((diff * norm.cdf(z)) + (sigma * norm.pdf(z)))

                best_acq = np.inf
                next_x = None
                
                # Multi-start acquisition optimization
                candidates = []
                # Local exploration via trust-region
                for _ in range(10):
                    x_c = np.clip(best_x_norm + np.random.normal(0, self.length, self.dim), 0, 1)
                    candidates.append(x_c)
                # Global exploration via random sampling
                for _ in range(5):
                    candidates.append(np.random.rand(self.dim))
                    
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            except:
                next_x = np.random.rand(self.dim)
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 0.5)
            else:
                self.length = np.clip(self.length * 0.9, 0.02, 0.5)
                
        return best_y, best_x_norm * (ub - lb) + lb