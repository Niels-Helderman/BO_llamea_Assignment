import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel: individual length scales for each dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        # Using a higher alpha to handle potential non-deterministic artifacts while remaining robust
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-7, normalize_y=True, n_restarts_optimizer=2)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial exploration with Sobol sequence
        init_size = 20
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        for i in range(init_size, self.budget):
            try:
                # Target output warping: target values are often log-normal or skewed
                # Log transformation helps focus on the tails
                y_transformed = np.log(y - np.min(y) + 1e-6)
                
                # Periodically fit kernel hyperparameters to balance overhead
                self.gp.fit(X, y_transformed if np.isfinite(y_transformed).all() else y)
                
                def objective(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.clip(sigma, 1e-12, None)
                    # EI calculation
                    y_best_tr = np.min(y_transformed)
                    Z = (y_best_tr - mu) / sigma
                    return -(sigma * (Z * norm.cdf(Z) + norm.pdf(Z)))

                # Global-Local hybrid search for acquisition maximum
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start gradient-based optimization
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except Exception:
                # Fallback: random search in space
                next_x = lb + np.random.rand(self.dim) * range_val

            # Execute observation
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x