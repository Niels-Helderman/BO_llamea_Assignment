import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel: allow different length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=False, n_restarts_optimizer=0)
        self.tr_size = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_y = np.min(y)
        best_x_norm = X[np.argmin(y)]
        
        for i in range(n_init, self.budget):
            # Log-transform target for better scaling in minimization
            y_min = np.min(y)
            y_trans = np.log(y - y_min + 1e-9)
            
            try:
                self.gp.fit(X, y_trans)
                fit_success = True
            except:
                fit_success = False

            def acq(x):
                if not fit_success: return np.random.rand()
                mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                # Acquisition: Expected Improvement on log-scale
                mu, std = mu[0], np.maximum(std[0], 1e-12)
                best_trans = np.min(y_trans)
                z = (best_trans - mu) / std
                return -((best_trans - mu) * norm.cdf(z) + std * norm.pdf(z))

            # Optimize acquisition
            next_x = None
            best_acq = np.inf
            # Multi-start: mix of trust-region local search and global random sampling
            candidates = [np.clip(best_x_norm + np.random.normal(0, self.tr_size, self.dim), 0, 1) for _ in range(10)]
            candidates += [np.random.rand(self.dim) for _ in range(5)]
            
            for x0 in candidates:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None: next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_size = min(self.tr_size * 1.1, 1.0)
            else:
                self.tr_size = max(self.tr_size * 0.9, 0.05)
                
        return float(best_y), best_x_norm * (ub - lb) + lb