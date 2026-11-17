import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_center = None
        self.tr_length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 2, min(self.budget // 4, 15))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                X_norm = (np.array(X_obs) - lb) / (ub - lb)
                y = np.array(y_obs)
                self.gp.fit(X_norm, y)
                
                # Adaptive UCB: beta decreases from 2.0 to 0.5 over the budget
                beta = 2.0 - (1.5 * i / self.budget)
                x = self._propose(X_norm, y, beta, lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            # Update trust region center to best point
            best_idx = np.argmin(y_obs)
            self.tr_center = (X_obs[best_idx] - lb) / (ub - lb)
            
        return float(np.min(y_obs)), np.array(X_obs[np.argmin(y_obs)])

    def _propose(self, X_norm, y, beta, lb, ub):
        def acq(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            return (mu - beta * sigma).flatten()

        best_x = None
        min_val = np.inf
        
        # Multi-start optimization
        for _ in range(8):
            # Local search around current best
            x0 = np.clip(self.tr_center + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < min_val:
                    min_val = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            best_x = np.random.uniform(0, 1, self.dim)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb