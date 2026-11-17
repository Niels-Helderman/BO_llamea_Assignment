import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel for individual feature sensitivity
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design using Latin Hypercube Sampling
        n_init = max(2 * self.dim + 1, self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        init_points = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
            else:
                x_norm = self._propose(lb, ub)
                
            x = x_norm * (ub - lb) + lb
            y = func(x)
            
            self.X_obs.append(x_norm)
            self.y_obs.append(y)
            
        best_idx = np.argmin(self.y_obs)
        return float(self.y_obs[best_idx]), self.X_obs[best_idx] * (ub - lb) + lb

    def _propose(self, lb, ub):
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        
        # Periodic refit to save cost, increase frequency as data grows
        fit = (len(self.X_obs) % 5 == 0) or (len(self.X_obs) < 20)
        if fit:
            self.gp.fit(X, y)
        
        y_min = np.min(y)
        
        def acquisition(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        # Multi-start optimization of the acquisition function
        best_acq = -np.inf
        target_x = np.random.rand(self.dim)
        
        # Candidates from random samples and best point
        starts = [self.X_obs[np.argmin(y)]] + [np.random.rand(self.dim) for _ in range(5)]
        
        for x0 in starts:
            try:
                res = minimize(acquisition, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    target_x = res.x
            except:
                continue
                
        return np.clip(target_x, 0, 1)