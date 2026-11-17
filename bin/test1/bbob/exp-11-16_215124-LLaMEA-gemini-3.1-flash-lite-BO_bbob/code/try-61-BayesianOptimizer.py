import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for better smoothness assumptions, WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial sampling using Latin Hypercube for space-filling
        init_size = min(max(2 * self.dim, 16), self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            try:
                # Periodic hyperparameter optimization
                self.gp.n_restarts_optimizer = 2 if (i % 50 == 0) else 0
                self.gp.fit(X, y)
                
                # Acquisition function: Log-EI (numerically stable)
                def neg_log_ei(x_norm):
                    mu, sigma = self.gp.predict((lb + x_norm * range_val).reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    # Log-EI helps prevent underflow
                    log_ei = np.log(np.maximum(diff * norm.cdf(z) + sigma * norm.pdf(z), 1e-20))
                    return -log_ei

                # Multi-start optimization of acquisition
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(10):
                    x0 = np.random.rand(self.dim)
                    res = minimize(neg_log_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except Exception:
                # Fallback to random point
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x