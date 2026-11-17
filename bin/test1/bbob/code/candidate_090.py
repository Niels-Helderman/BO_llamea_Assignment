import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use a stationary Matern kernel for general smoothness handling
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial exploration: use a fixed, slightly larger LHS to improve early GP quality
        init_size = min(max(4 * self.dim, 24), self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        # Trust region radius normalized to [0, 1]
        tr_len = 0.4
        
        for i in range(init_size, self.budget):
            try:
                # Periodic hyperparameter tuning to keep speed while maintaining GP quality
                self.gp.n_restarts_optimizer = 3 if (i % 30 == 0) else 0
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x_norm = np.clip(x_norm, 0, 1)
                    mu, sigma = self.gp.predict((lb + x_norm * range_val).reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start optimization around the current best to focus exploitation
                best_ei = -np.inf
                best_norm_x = (best_x - lb) / range_val
                
                for _ in range(12):
                    # Sample starting points inside the trust region
                    x0 = np.clip(best_norm_x + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x = lb + res.x * range_val
            except:
                # Robust fallback for acquisition optimization failures
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive logic for TR
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len = max(0.05, tr_len * 0.8)
                
        return best_y, best_x