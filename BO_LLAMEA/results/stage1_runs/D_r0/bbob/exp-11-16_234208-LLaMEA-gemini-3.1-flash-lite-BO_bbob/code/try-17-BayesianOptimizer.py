import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to learn dimension-wise importance
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.4
        self.min_tr = 0.05
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim, min(self.budget // 4, 15))
        
        X_obs = []
        y_obs = []
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
            else:
                x_norm = self._propose(np.array(X_obs), np.array(y_obs))
            
            x = x_norm * (ub - lb) + lb
            y = func(x)
            
            X_obs.append(x_norm)
            y_obs.append(y)
            
            # Adaptive trust region update
            if i >= n_init:
                best_y = np.min(y_obs[:-1])
                if y < best_y:
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
                    
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), X_obs[best_idx] * (ub - lb) + lb

    def _propose(self, X, y):
        # Log-transform targets for better GP stability on skewed objective landscapes
        y_log = np.log(np.array(y) - np.min(y) + 1e-6)
        self.gp.fit(X, y_log)
        
        best_x = X[np.argmin(y)]
        y_min = np.min(y)
        
        def acq(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, std = self.gp.predict(x_norm, return_std=True)
            mu = np.exp(mu) + y_min
            # EI acquisition
            imp = y_min - mu
            z = imp / (std + 1e-9)
            return -(imp * norm.cdf(z) + std * norm.pdf(z))

        # Multi-start optimization in Trust Region
        best_val = np.inf
        best_cand = np.random.rand(self.dim)
        
        for _ in range(8):
            x0 = np.clip(best_x + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_val:
                    best_val = res.fun
                    best_cand = res.x
            except:
                continue
        return best_cand