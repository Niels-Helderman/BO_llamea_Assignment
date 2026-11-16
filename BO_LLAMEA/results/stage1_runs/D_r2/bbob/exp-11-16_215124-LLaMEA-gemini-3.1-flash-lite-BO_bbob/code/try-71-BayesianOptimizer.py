import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0.0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # LHS initialization
        init_size = min(max(2 * self.dim, 10), self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Refit kernel periodically
                self.gp.n_restarts_optimizer = 3 if i % 30 == 0 else 0
                self.gp.fit(X, y)
                
                def ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                best_ei = -np.inf
                best_candidate = lb + np.random.rand(self.dim) * range_val
                tr_center = (best_x - lb) / range_val
                
                # Multi-start acquisition in trust region
                for _ in range(10):
                    x0 = np.clip(tr_center + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                    res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        best_candidate = lb + res.x * range_val
                next_x = best_candidate
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len *= 0.8
                if tr_len < 0.01: tr_len = 0.5 # Reset trust region if too small
                
        return best_y, best_x