import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is robust for black-box optimization
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Use Sobol sequence for initial coverage
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                # Calculate dynamic beta for UCB: explore more early, exploit late
                progress = i / self.budget
                beta = 2.0 * (1.0 - progress) + 0.5
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, beta)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, beta):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        
        def ucb(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            return (mu - beta * sigma).flatten()

        # Multi-start optimization of the acquisition function
        best_val = np.inf
        best_x = None
        
        # Candidates from random sampling + local optimization
        n_candidates = 20
        candidates = np.random.rand(n_candidates, self.dim)
        
        for x0 in candidates:
            try:
                res = minimize(ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
                
        if best_x is None:
            best_x = np.random.rand(self.dim)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb