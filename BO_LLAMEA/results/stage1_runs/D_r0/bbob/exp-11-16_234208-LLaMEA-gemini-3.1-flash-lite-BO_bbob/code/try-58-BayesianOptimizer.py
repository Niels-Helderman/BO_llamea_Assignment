import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize, differential_evolution

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.05
        self.max_tr = 0.8

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
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
            if i >= n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.1)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.9)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        y_min = np.min(y)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-10)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Multi-strategy acquisition maximization
        best_x = None
        best_val = np.inf
        
        # 1. Differential Evolution for global acquisition exploration
        try:
            res = differential_evolution(acq, [(0, 1)] * self.dim, tol=1e-3, popsize=10, maxiter=20)
            best_x, best_val = res.x, res.fun
        except: pass
        
        # 2. Refine with L-BFGS-B starting from the current best point
        best_known_norm = X_norm[np.argmin(y)]
        try:
            res_local = minimize(acq, best_known_norm, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-5)
            if res_local.fun < best_val:
                best_x, best_val = res_local.x, res_local.fun
        except: pass
        
        if best_x is None:
            return np.random.uniform(lb, ub)
        
        return np.clip(best_x, 0, 1) * (ub - lb) + lb