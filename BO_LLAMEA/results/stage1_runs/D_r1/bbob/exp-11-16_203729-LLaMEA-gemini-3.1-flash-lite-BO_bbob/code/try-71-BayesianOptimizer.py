import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a more robust kernel initialization and fixed jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds='fixed') + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initial design density for better early-stage surrogate fitting
        n_init = max(2 * self.dim + 8, 20)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                # Hyperparameters are kept stable by setting fixed length_scale_bounds
            except:
                pass # Continue with previous surrogate if fit fails
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # Log-EI to prevent underflow and improve numerical stability
                diff = best_y - mu
                z = diff / sigma
                log_ei = np.log(np.maximum(diff, 1e-12)) + norm.logcdf(z) + np.log(1 + (sigma/diff)*norm.pdf(z)/norm.cdf(z) + 1e-12)
                return -log_ei

            # Adaptive search: mix trust-region and global space exploration
            best_acq = np.inf
            next_x = None
            
            candidates = []
            # Local bias
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            for _ in range(10):
                candidates.append(np.random.uniform(lb_tr, ub_tr))
            # Global exploration
            for _ in range(5):
                candidates.append(np.random.uniform(0, 1, self.dim))
                
            for x0 in candidates:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.9, 0.05, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb