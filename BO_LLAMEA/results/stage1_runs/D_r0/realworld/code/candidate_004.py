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
        # Use Matern 5/2, normalize_y handles scale issues. 
        # WhiteKernel is critical for robustness against objective noise.
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=3
        )

    def _get_ei(self, x, y_best):
        x = x.reshape(1, -1)
        with np.errstate(divide='ignore'):
            mu, sigma = self.gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            imp = y_best - mu
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei[0]

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        # Initial phase: Latin Hypercube-like spacing for 1.5*dim
        init_size = max(2 * self.dim, 6)
        
        for i in range(self.budget):
            if i < init_size:
                x = np.random.uniform(lb, ub)
            else:
                y_best = np.min(self.y_hist)
                # Multi-start acquisition optimization to avoid local optima
                best_ei = np.inf
                best_x_acq = None
                
                # Check current best and several random candidates
                candidates = [self.X_hist[np.argmin(self.y_hist)]]
                for _ in range(5):
                    candidates.append(np.random.uniform(lb, ub))
                
                for cand in candidates:
                    res = minimize(
                        lambda x_n: self._get_ei(x_n * (ub - lb) + lb, y_best),
                        x0=(cand - lb) / (ub - lb),
                        bounds=[(0, 1)] * self.dim,
                        method='L-BFGS-B'
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x_acq = res.x * (ub - lb) + lb
                x = best_x_acq

            x = np.clip(x, lb, ub)
            y = float(func(x))
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Retrain GP surrogate periodically
            if len(self.X_hist) >= 3:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except Exception:
                    pass 
                    
        return float(best_y), best_x