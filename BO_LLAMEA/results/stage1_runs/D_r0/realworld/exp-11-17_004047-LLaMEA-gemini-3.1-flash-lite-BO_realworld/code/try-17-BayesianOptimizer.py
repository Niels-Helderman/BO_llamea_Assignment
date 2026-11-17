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
        # Use ARD Matern kernel for high-dim flexibility, log-transform targets y to handle scale
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)
        self.tr_size = 0.5 

    def _get_ei(self, x_norm, gp, y_best, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-12, None)
        # Using Expected Improvement on the target function
        imp = y_best - mu
        z = imp / sigma
        return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # Latin Hypercube-like sampling
                x = lb + (ub - lb) * (np.random.rand(self.dim))
            else:
                y_best = np.min(self.y_hist)
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                # Trust region search to prevent exploration of poor zones
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                try:
                    res = minimize(
                        lambda x_n: self._get_ei(x_n, self.gp, y_best, lb, ub),
                        x0=np.clip(x_inc_norm, tr_lb, tr_ub),
                        bounds=list(zip(tr_lb, tr_ub)),
                        method='L-BFGS-B'
                    )
                    x = res.x * (ub - lb) + lb
                    if res.fun < -1e-6:
                        self.tr_size = min(1.0, self.tr_size * 1.1)
                    else:
                        self.tr_size = max(0.05, self.tr_size * 0.8)
                except:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            y = float(func(x))
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Retrain GP with robust error handling
            if len(self.X_hist) >= 5:
                try:
                    X_np = np.array(self.X_hist)
                    y_np = np.array(self.y_hist)
                    # Log-transform if data allows to stabilize objective scale
                    y_transformed = np.log(y_np - np.min(y_np) + 1e-6) if np.min(y_np) < 0 else y_np
                    self.gp.fit(X_np, y_transformed)
                except:
                    pass
                    
        return float(best_y), np.array(best_x)