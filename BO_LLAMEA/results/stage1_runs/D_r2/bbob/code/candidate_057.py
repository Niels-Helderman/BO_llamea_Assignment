import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Using ARD (Automatic Relevance Determination) Matern kernel to capture dimension-specific sensitivities
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Latin Hypercube Sampling for initial space filling
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            try:
                # Periodic hyperparameter optimization with limited restarts for efficiency
                self.gp.n_restarts_optimizer = 3 if (i % 50 == 0) else 0
                self.gp.fit(X, y)
                
                def ei_acquisition(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start acquisition optimization (Global + Local)
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                starts = [np.random.rand(self.dim) for _ in range(5)]
                starts.append((best_x - lb) / range_val) # Seed one start at current best
                
                for x0 in starts:
                    res = minimize(ei_acquisition, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Robust fallback for numerical instability
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            
            # Update surrogate data
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return float(best_y), best_x