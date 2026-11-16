import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 with ARD and WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-12)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Enhanced initialization: Latin Hypercube Sampling for better early space filling
        init_size = min(max(2 * self.dim + 2, 12), self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # GP refitting: reuse kernel params to speed up
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x_norm = None
                
                # Multi-start acquisition optimization within Adaptive Trust Region
                tr_center = (best_x - lb) / range_val
                candidates = []
                # Add random points and points around current best
                for _ in range(12):
                    start_pt = np.clip(tr_center + np.random.normal(0, tr_len, self.dim), 0, 1)
                    res = minimize(get_ei, start_pt, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    candidates.append((res.fun, res.x))
                
                best_ei, next_x_norm = min(candidates, key=lambda x: x[0])
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random search if GP fit or optimization fails
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Trust region update logic
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len = max(0.05, tr_len * 0.8)
                
        return float(best_y), best_x