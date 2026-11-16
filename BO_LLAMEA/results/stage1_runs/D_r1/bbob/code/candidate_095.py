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
        # Using alpha=1e-6 to handle potential numerical instability in deterministic functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 4, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Only perform full hyperparameter optimization every 10 iterations to save cost
            if (i - n_init) % 10 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            
            try:
                self.gp.fit(X, y)
            except:
                pass # Fallback to existing GP state if fit fails
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI is more numerically stable than raw EI
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            best_acq = np.inf
            next_x = None
            
            candidates = []
            # Local search around current best (exploitation)
            for _ in range(12):
                c = np.clip(best_x_norm + np.random.normal(0, self.length, self.dim), 0, 1)
                candidates.append(c)
            # Global exploration
            for _ in range(4):
                candidates.append(np.random.rand(self.dim))
                
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
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