import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # Robust kernel: Constant scale * Matern 5/2 + WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.tr_size = 0.5

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)
        
        # Increased initial sampling to accelerate early convergence
        n_init = max(2 * self.dim + 4, 16)
        
        for i in range(self.budget):
            if i < n_init:
                # Use quasi-random/dispersed initialization
                x = lb + (ub - lb) * ((np.arange(self.dim) * 0.618 + i * 0.382) % 1.0)
            else:
                y_best = np.min(self.y_hist)
                x_inc = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                # Optimize EI with multi-start local optimization
                res = minimize(
                    lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                    x0=x_inc_norm,
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Adaptive TR sizing logic
                if res.fun < -1e-6:
                    self.tr_size = min(0.5, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.05, self.tr_size * 0.7)
            
            x = np.clip(x, lb, ub)
            
            # Distance-based duplicate checking to keep GP well-conditioned
            if len(self.X_hist) > 0 and np.min(np.linalg.norm(np.array(self.X_hist) - x, axis=1)) < 1e-7:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Periodical GP update: schedule fitting every step after n_init
            if len(self.X_hist) >= 5:
                try:
                    self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=1)
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)