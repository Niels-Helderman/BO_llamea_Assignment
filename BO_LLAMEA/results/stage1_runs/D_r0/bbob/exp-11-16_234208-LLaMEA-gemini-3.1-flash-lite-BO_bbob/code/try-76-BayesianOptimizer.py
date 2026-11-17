import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel for deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=2)
        # Adaptive LCB parameters
        self.beta_init = 2.0
        self.beta_final = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial Design: Sobol sequence for better space coverage than random
        n_init = max(2, min(self.budget // 5, 2 * self.dim + 2))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, i)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, iteration):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        # Adaptive LCB schedule
        beta = self.beta_init - (self.beta_init - self.beta_final) * (iteration / self.budget)
        
        def lcb(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        # Multi-start optimization of acquisition function
        best_x_norm = None
        min_acq = np.inf
        
        # Start from best observation and random points
        candidates = [X_norm[np.argmin(y)]] + [np.random.rand(self.dim) for _ in range(8)]
        
        for x0 in candidates:
            try:
                res = minimize(lcb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < min_acq:
                    min_acq = res.fun
                    best_x_norm = res.x
            except:
                continue
        
        if best_x_norm is None:
            return np.random.uniform(lb, ub)
            
        return np.clip(best_x_norm, 0, 1) * (ub - lb) + lb