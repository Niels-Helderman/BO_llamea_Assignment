import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Jitter added to diagonal to prevent singular matrix issues during updates
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial exploration: use a smaller balanced Sobol design for faster start
        n_init = min(12, self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Refit GP periodically
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                # If fitting fails, sample randomly
                next_x = np.random.rand(self.dim)
            else:
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    # Use Log-EI to handle numerical stability and steep gradients
                    log_ei = np.log(imp + 1e-12) + norm.logcdf(z) + np.log(1 + (sigma / (imp + 1e-12)) * norm.pdf(z) / (norm.cdf(z) + 1e-12))
                    return -log_ei

                best_acq = np.inf
                next_x = None
                
                # Multi-start optimization inside and outside current trust region
                starts = [best_x_norm] + [np.clip(best_x_norm + np.random.normal(0, self.tr_size, self.dim), 0, 1) for _ in range(6)] + [np.random.rand(self.dim) for _ in range(3)]
                
                for x0 in starts:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
                
                if next_x is None: next_x = np.random.rand(self.dim)

            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update best and dynamic trust-region size
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_size = min(self.tr_size * 1.1, 1.0)
            else:
                self.tr_size = max(self.tr_size * 0.9, 0.05)
                
        return best_y, best_x_norm * (ub - lb) + lb