import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to learn feature importance in 6D
        self.kernel = Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Space-filling LHS initialization
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        # Local search radius for trust region
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Update surrogate
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    mu, sigma = self.gp.predict((lb + x_norm * range_val).reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start acquisition optimization
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                tr_center = (best_x - lb) / range_val
                
                for _ in range(10):
                    # Hybrid: 50% global exploration, 50% local refinement
                    if np.random.rand() < 0.5:
                        x0 = np.random.rand(self.dim)
                        bounds = [(0, 1) for _ in range(self.dim)]
                    else:
                        x0 = np.clip(tr_center + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                        bounds = list(zip(np.clip(tr_center - tr_len, 0, 1), np.clip(tr_center + tr_len, 0, 1)))
                    
                    res = minimize(get_ei, x0, bounds=bounds, method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive logic
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len = max(0.05, tr_len * 0.9)
                
        return best_y, best_x