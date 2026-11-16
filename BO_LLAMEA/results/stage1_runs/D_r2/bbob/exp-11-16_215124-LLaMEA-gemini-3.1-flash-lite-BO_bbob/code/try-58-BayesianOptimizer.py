import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space filling than pure random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = 2 * self.dim + 2
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Refit kernel hyperparams every 30 steps to avoid excessive overhead
                if i % 30 == 0:
                    self.gp.n_restarts_optimizer = 3
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = np.clip(x_norm, 0, 1)
                    mu, sigma = self.gp.predict((lb + x * range_val).reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start acquisition in trust region
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                tr_center = (best_x - lb) / range_val
                
                for _ in range(5):
                    x0 = np.clip(tr_center + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei, next_x_norm = -res.fun, res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.05, tr_len * 0.95)
                
        return best_y, best_x