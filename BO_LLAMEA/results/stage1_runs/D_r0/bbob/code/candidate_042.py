import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for general smoothness, WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 
        self.min_tr = 0.01
        self.max_tr = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initial design to capture landscape features early (approx 20% of budget)
        n_init = max(2 * self.dim + 2, int(0.2 * self.budget))
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
            
            # Adaptive trust region: shrink/expand based on success
            if i > n_init:
                if y < np.min(y_obs[:-1]):
                    self.tr_length = min(self.max_tr, self.tr_length * 1.2)
                else:
                    self.tr_length = max(self.min_tr, self.tr_length * 0.8)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        
        try:
            self.gp.fit(X_norm, y)
        except:
            return np.random.uniform(lb, ub)
        
        y_min = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        def objective(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-12)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        # Multi-start acquisition optimization
        best_acq = -1.0
        target_x = None
        
        # Candidate generation: mix local search and random sampling
        starts = [best_x_norm] + [np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1) for _ in range(8)]
        
        for x0 in starts:
            try:
                res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except:
                continue
        
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb