import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel with small noise to handle deterministic objective noise
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol
        n_init = min(self.budget // 4, 2 * self.dim + 6)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically re-fit GP kernel to maintain surrogate quality
            self.gp.fit(X, y)
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                with np.errstate(divide='ignore'):
                    z = (best_y - mu) / sigma
                    ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Multi-start acquisition optimization using localized search
            best_acq_val = np.inf
            next_x = None
            
            # Combine local trust region and global exploration seeds
            candidates = [best_x_norm + np.random.uniform(-self.tr_length, self.tr_length, self.dim) 
                          for _ in range(12)]
            candidates += [np.random.rand(self.dim) for _ in range(4)]
            candidates = np.clip(candidates, 0, 1)
                
            for x0 in candidates:
                try:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq_val:
                        best_acq_val, next_x = res.fun, res.x
                except: continue
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            # Execute objective call
            try:
                val = func(next_x * (ub - lb) + lb)
            except:
                val = np.inf # Fallback for numerical instability
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive trust region adjustment
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_length = np.clip(self.tr_length * 1.1, 0.05, 0.8)
            else:
                self.tr_length = np.clip(self.tr_length * 0.9, 0.05, 0.8)
                
        return best_y, best_x_norm * (ub - lb) + lb