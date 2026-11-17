import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.02
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        # Use augmented LHS for better initial spread
        n_init = max(2 * self.dim + 2, min(self.budget // 4, 15))
        sampler = qmc.LatinHypercube(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub, best_y)
            
            try:
                y = float(func(x))
            except:
                y = 1e10 # Fallback for black-box failure
            
            X_obs.append(x)
            y_obs.append(y)
            if y < best_y:
                best_y = y
                if i >= n_init: self.tr_length = min(self.max_tr, self.tr_length * 1.2)
            elif i >= n_init:
                self.tr_length = max(self.min_tr, self.tr_length * 0.7)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub, y_min):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        best_x_norm = X_norm[np.argmin(y)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-10)
                return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        best_acq = np.inf
        target_x = None
        
        # Multi-start optimization within Trust Region
        starts = [best_x_norm + np.random.uniform(-self.tr_length, self.tr_length, self.dim) for _ in range(15)]
        starts.append(np.random.rand(self.dim)) # Global component
        
        for x0 in starts:
            x0 = np.clip(x0, 0, 1)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, target_x = res.fun, res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb