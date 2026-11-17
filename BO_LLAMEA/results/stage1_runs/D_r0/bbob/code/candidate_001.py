import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern kernel for flexibility, add small jitter for numerical stability
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        X_obs = []
        y_obs = []
        
        best_y = np.inf
        best_x = None

        # Initialization: Use a small Latin Hypercube-like spread or random points
        # 10% of budget or at least dim+2 for GP stability
        n_init = max(self.dim + 2, int(0.1 * self.budget))
        
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                # Fit GP periodically
                try:
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                    x = self._propose_next(lb, ub, range_val, y_obs)
                except:
                    x = np.random.uniform(lb, ub)

            y = func(x)
            
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                
        return float(best_y), np.array(best_x)

    def _propose_next(self, lb, ub, range_val, y_obs):
        best_y = min(y_obs)
        
        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            with np.errstate(divide='ignore'):
                z = (best_y - mu) / sigma
                ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei[0]

        # Multi-start acquisition optimization
        best_ei = -np.inf
        proposal = None
        for _ in range(5):
            x0 = np.random.uniform(lb, ub)
            res = minimize(objective, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_ei:
                best_ei = -res.fun
                proposal = res.x
        
        return proposal if proposal is not None else np.random.uniform(lb, ub)