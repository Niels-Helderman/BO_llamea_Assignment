import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD kernel to allow different length scales for each dimension
        # Log-transforming target values can stabilize variance in non-stationary BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial sampling using Latin Hypercube
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            # Periodically re-optimize hyperparameters to adapt to data
            if i % 15 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
                
            try:
                # Fit GP on log-transformed data to improve numerical range handling
                y_transformed = np.log(y - np.min(y) + 1e-6)
                self.gp.fit(X, y_transformed)
                
                def get_ei(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu_log, std_log = self.gp.predict(x, return_std=True)
                    # Expected Improvement on original scale via delta-method approximation
                    mu = np.exp(mu_log) 
                    sigma = std_log * mu
                    
                    imp = best_y - mu
                    Z = imp / (sigma + 1e-12)
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start acquisition optimization
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(10):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x