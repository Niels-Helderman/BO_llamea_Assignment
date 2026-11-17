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
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                beta = 2.0 * np.log(i + 1)
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, beta)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, beta):
        X_norm = (X - lb) / (ub - lb)
        y_centered = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        try:
            self.gp.fit(X_norm, y_centered)
        except:
            return np.random.uniform(lb, ub)

        def objective(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            return mu - np.sqrt(beta) * sigma

        best_x = None
        min_val = np.inf
        
        # Multi-start optimization for acquisition
        starts = np.random.rand(10, self.dim)
        for x0 in starts:
            res = minimize(objective, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                best_x = res.x
                
        if best_x is None:
            best_x = np.random.rand(self.dim)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb