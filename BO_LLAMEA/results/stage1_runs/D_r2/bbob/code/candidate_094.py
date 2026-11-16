import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to capture different feature sensitivities
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Space-filling LHS initialization
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(y)].copy()
        best_y = float(np.min(y))
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters periodically
                if i % 10 == 0:
                    self.gp.n_restarts_optimizer = 3
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                # EI Acquisition
                def get_ei(x_norm):
                    mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start acquisition optimization
                center = (best_x - lb) / range_val
                for _ in range(12):
                    start = np.clip(center + np.random.normal(0, tr_len, self.dim), 0, 1)
                    res = minimize(get_ei, start, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
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