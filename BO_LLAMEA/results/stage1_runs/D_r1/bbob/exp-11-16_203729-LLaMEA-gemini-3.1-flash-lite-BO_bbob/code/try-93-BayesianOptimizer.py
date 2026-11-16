import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a more robust kernel setup with WhiteKernel for numerical stability on deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol sequence for better space filling
        n_init = max(2 * self.dim + 4, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters, but use existing to save cost as data grows
            if i % 5 == 0 or i < n_init + 5:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            # Log-EI acquisition for better numerical stability in exploitation
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (best_y - mu) / sigma
                    log_ei_val = np.log(sigma) + norm.logcdf(z) + np.log1p(np.exp(norm.logpdf(z) - norm.logcdf(z) - np.log(sigma + 1e-16) + np.log(best_y - mu + 1e-16)))
                return -np.exp(np.nan_to_num(log_ei_val))

            best_acq = np.inf
            next_x = None
            
            # Hybrid search: Local trust-region biased search + Global random exploration
            candidates = []
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            for _ in range(12):
                candidates.append(np.random.uniform(lb_tr, ub_tr))
            for _ in range(4):
                candidates.append(np.random.uniform(0, 1, self.dim))
                
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback for acquisition failure
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive trust-region update
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 0.5)
            else:
                self.length = np.clip(self.length * 0.9, 0.05, 0.5)
                
        return best_y, best_x_norm * (ub - lb) + lb