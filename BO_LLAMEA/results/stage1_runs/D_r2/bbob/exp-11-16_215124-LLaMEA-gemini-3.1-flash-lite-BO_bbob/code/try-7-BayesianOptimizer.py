import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Initial space-filling (LHS-like)
        n_init = max(2 * self.dim + 2, 12)
        X, y = [], []
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            X.append(x)
            y.append(func(x))
        X, y = np.array(X), np.array(y)
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        # Simple CMA-ES parameters for acquisition optimization
        sigma = 0.3
        pop_size = 4 + int(3 * np.log(self.dim))
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                
                # EI acquisition function
                def get_ei(x_norm):
                    x = lb + x_norm * scale
                    mu, sigma_gp = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma_gp = np.maximum(sigma_gp, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma_gp
                    return -(imp * norm.cdf(Z) + sigma_gp * norm.pdf(Z))

                # CMA-ES style sampling to find acquisition maximum
                best_ei_val = -np.inf
                best_candidate = None
                
                for _ in range(15): # Local search/Optimization steps
                    candidates = np.random.normal(loc=(best_x - lb)/scale, scale=sigma, size=(pop_size, self.dim))
                    candidates = np.clip(candidates, 0, 1)
                    
                    ei_vals = [get_ei(c) for c in candidates]
                    idx = np.argmin(ei_vals)
                    if -ei_vals[idx] > best_ei_val:
                        best_ei_val = -ei_vals[idx]
                        best_candidate = candidates[idx]
                
                next_x = lb + best_candidate * scale
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                sigma = max(0.05, sigma * 0.95)
            else:
                sigma = min(0.5, sigma * 1.05)
                
        return float(best_y), best_x