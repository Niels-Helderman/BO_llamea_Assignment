import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is robust for black-box optimization
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5
        self.min_tr = 0.05
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling
        n_init = max(2 * self.dim, min(self.budget // 4, 15))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y, best_x = float('inf'), None
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y, best_x = float(y), x
            
            # Simple adaptive TR update
            if i > n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
                    
        return best_y, best_x

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-12)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        # Optimize acquisition using multi-start L-BFGS-B
        best_acq = np.inf
        target_x = None
        for _ in range(8):
            start_x = np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1)
            res = minimize(acq, start_x, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
            if res.fun < best_acq:
                best_acq = res.fun
                target_x = res.x
        
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb