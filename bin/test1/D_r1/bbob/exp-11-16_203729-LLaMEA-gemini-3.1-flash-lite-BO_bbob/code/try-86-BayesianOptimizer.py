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
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialize with Sobol sequence for better space-filling
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_norm])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X_norm[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically re-fit GP
            self.gp.fit(X_norm, y)
            
            # Log-EI for numerical stability in late-stage exploitation
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI calculation: log(EI) = log(diff * Phi(z) + sigma * phi(z))
                # Computed using log-sum-exp trick style logic if needed, but here standard EI
                return -((diff * norm.cdf(z) + sigma * norm.pdf(z)))

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Focus exploration near current best via trust region
            candidates = []
            for _ in range(10):
                candidates.append(np.clip(best_x_norm + np.random.normal(0, self.length, self.dim), 0, 1))
            for _ in range(5):
                candidates.append(np.random.rand(self.dim))
                
            for x0 in candidates:
                try:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
                except: continue
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X_norm = np.vstack([X_norm, next_x])
            y = np.append(y, val)
            
            # Adaptive trust-region update
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 0.5)
            else:
                self.length = np.clip(self.length * 0.9, 0.05, 0.5)
                
        return best_y, best_x_norm * (ub - lb) + lb