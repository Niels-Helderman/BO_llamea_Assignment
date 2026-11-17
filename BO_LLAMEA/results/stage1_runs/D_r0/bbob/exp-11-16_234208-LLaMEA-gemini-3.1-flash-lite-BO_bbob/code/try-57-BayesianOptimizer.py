import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel for deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial Design: Sobol sequence for better space filling
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                # Periodic refitting to maintain performance
                if i % 5 == 0:
                    self.gp.fit(X_norm, np.array(y_obs))
                
                # Adaptive LCB: beta decays as we get more samples to focus on exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                x = self._propose(X_norm, np.array(y_obs), beta, lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            X_norm = (np.array(X_obs) - lb) / (ub - lb)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X_norm, y, beta, lb, ub):
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y_scaled = (y - y_mean) / y_std
        
        self.gp.fit(X_norm, y_scaled)
        
        # LCB Acquisition Function
        def lcb(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return (mu - beta * std)[0]

        # Multi-start optimization
        best_x, min_acq = None, np.inf
        seeds = np.random.uniform(0, 1, (10, self.dim))
        # Add current best as a starting point for local exploitation
        seeds = np.vstack([seeds, X_norm[np.argmin(y)]])
        
        for x0 in seeds:
            res = minimize(lcb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
            if res.fun < min_acq:
                min_acq = res.fun
                best_x = res.x
        
        if best_x is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(best_x, 0, 1) * (ub - lb) + lb