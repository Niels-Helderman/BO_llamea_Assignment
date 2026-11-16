import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD kernel: length_scale is a vector of size (dim,)
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial exploration using Latin Hypercube Sampling
        init_size = max(2 * self.dim + 2, 16)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            # Fit model with periodic hyperparameter optimization
            if i % 15 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
                
            try:
                self.gp.fit(X, y)
                
                def ei(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start acquisition optimization
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                for _ in range(10):
                    x0 = np.random.rand(self.dim)
                    res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            
            # Update history
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x