import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 4, 12)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        def get_log_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e6)
            z = (y_min - mu) / sigma
            # Return log of EI for numerical stability
            return np.log(np.clip((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z), 1e-20, None))

        for i in range(n_init, self.budget):
            if i % 5 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            y_min = np.min(y)
            
            # Hybrid selection: 80% GP-EI, 20% Local mutation (CMA-like)
            if np.random.rand() > 0.2:
                best_acq = -np.inf
                next_x = np.random.uniform(0, 1, self.dim)
                for _ in range(8):
                    x0 = np.random.uniform(0, 1, self.dim)
                    res = minimize(lambda x: -get_log_ei(x, self.gp, y_min), x0, 
                                   bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if -res.fun > best_acq:
                        best_acq = -res.fun
                        next_x = res.x
            else:
                # Local mutation around current best to escape GP plateaus
                step_size = 0.1 * (0.95**(i/10))
                next_x = np.clip(best_x_norm + np.random.normal(0, step_size, self.dim), 0, 1)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                
        return best_y, best_x_norm * (ub - lb) + lb