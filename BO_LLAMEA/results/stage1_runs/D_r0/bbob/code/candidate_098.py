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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.rng = np.random.RandomState(42)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 4)
        
        # Sobol sequence for better space coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        X_init = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        for i in range(self.budget):
            if i < n_init:
                x_norm = X_init[i]
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs))
            
            x = x_norm * (ub - lb) + lb
            y = float(func(x))
            X_obs.append(x_norm)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), X_obs[best_idx] * (ub - lb) + lb

    def _propose(self, X, y):
        # Adaptive UCB: beta schedule to favor exploration early, exploitation late
        t = len(y)
        beta = 2.0 * np.log(t + 1)
        
        self.gp.fit(X, y)
        y_min = np.min(y)
        
        def ucb(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        # Multi-start optimization for acquisition function
        best_val = np.inf
        best_x = X[np.argmin(y)]
        
        # Random candidates + best current point
        candidates = [np.random.rand(self.dim) for _ in range(20)]
        candidates.append(X[np.argmin(y)])
        
        for x0 in candidates:
            try:
                res = minimize(ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
                
        return np.clip(best_x, 0, 1)