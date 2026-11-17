import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small noise term for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.fitted_count = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Initial points using Latin Hypercube-like spacing
        n_init = max(2 * self.dim + 2, self.budget // 5)
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * (np.random.rand(self.dim) + np.arange(self.dim)) / (self.dim + 1)
                x = np.clip(x, lb, ub)
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, i)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, step):
        # Normalize inputs
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        # Beta schedule for UCB (Exploration -> Exploitation)
        beta = 2.0 * np.log(step + 1)
        
        def ucb_obj(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            return -(mu - beta * sigma).flatten()

        # Multi-start acquisition optimization
        best_acq = -np.inf
        target_x = None
        for _ in range(10):
            x0 = np.random.rand(self.dim)
            try:
                res = minimize(ucb_obj, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except:
                continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(target_x, 0, 1) * (ub - lb) + lb