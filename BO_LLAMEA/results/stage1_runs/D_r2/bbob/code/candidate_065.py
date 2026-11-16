import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for better handling of non-smooth BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space filling than pure random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        tr_len = 0.5
        for i in range(init_size, self.budget):
            # Fit GP with periodic hyperparameter optimization
            self.gp.n_restarts_optimizer = 2 if (i % 30 == 0) else 0
            try:
                self.gp.fit(X, y)
                
                # Log-EI to improve numerical stability and gradient-based optimization
                def get_neg_log_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.clip(sigma, 1e-12, None)
                    diff = best_y - mu
                    z = diff / sigma
                    # Log-EI calculation
                    log_ei = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z) + \
                             np.log(1 + (sigma / np.maximum(diff, 1e-20)) * norm.pdf(z) / norm.cdf(z))
                    return -log_ei

                # Multi-start acquisition optimization constrained by adaptive trust region
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                tr_center = (best_x - lb) / range_val
                
                for _ in range(10):
                    x0 = np.clip(tr_center + np.random.normal(0, tr_len / 2, self.dim), 0, 1)
                    res = minimize(get_neg_log_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len = max(0.05, tr_len * 0.9)
                
        return best_y, best_x