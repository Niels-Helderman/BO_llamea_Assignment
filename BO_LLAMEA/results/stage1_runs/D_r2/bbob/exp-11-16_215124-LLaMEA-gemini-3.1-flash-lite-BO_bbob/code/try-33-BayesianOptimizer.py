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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial sampling
        init_size = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            try:
                # Update surrogate
                self.gp.fit(X, y)
                
                # EI Acquisition
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start optimization of EI
                best_ei = -np.inf
                next_x_norm = None
                
                # Propose candidates: 5 global random, 5 local around best_x
                starts = np.random.rand(10, self.dim)
                starts[5:] = (best_x - lb) / range_val + np.random.normal(0, 0.1, (5, self.dim))
                starts = np.clip(starts, 0, 1)
                
                for x0 in starts:
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x