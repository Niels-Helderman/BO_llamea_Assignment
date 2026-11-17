import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Constant * Matern + jitter for deterministic functions
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def _get_ei(self, x, gp, y_best):
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        diff = y_best - mu
        with np.errstate(divide='ignore'):
            z = diff / sigma
            from scipy.stats import norm
            ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei.item()

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_hist, y_hist = [], []
        
        # Initial sampling: Latin Hypercube-like spread using Sobol-sequence-inspired random
        n_init = min(12, self.budget // 4)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            y = func(x)
            X_hist.append(x)
            y_hist.append(y)
            
        best_idx = np.argmin(y_hist)
        best_y, best_x = y_hist[best_idx], X_hist[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_hist)
            y = np.array(y_hist)
            
            # Periodically refit hyperparameters
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 1
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X, y)
                y_best = np.min(y)
                
                # Acquisition optimization via multi-start L-BFGS-B
                best_acq = np.inf
                candidate = None
                for _ in range(5):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(self._get_ei, x0, args=(self.gp, y_best), 
                                   bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        candidate = res.x
            except:
                candidate = np.random.uniform(lb, ub)
            
            x_next = candidate
            y_next = func(x_next)
            
            X_hist.append(x_next)
            y_hist.append(y_next)
            
            if y_next < best_y:
                best_y, best_x = y_next, x_next
                
        return float(best_y), best_x