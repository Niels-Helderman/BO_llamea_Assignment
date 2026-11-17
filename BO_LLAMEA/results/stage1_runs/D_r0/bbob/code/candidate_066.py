import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize, differential_evolution

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel setup for BBOB functions
        self.kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for initial exploration
        n_init = min(self.budget, 2 * self.dim + 5)
        sampler = qmc.LatinHypercube(d=self.dim)
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
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        try:
            self.gp.fit(X_norm, y)
        except:
            return np.random.uniform(lb, ub)
        
        y_min = np.min(y)
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-9)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei # Minimize negative EI

        # Hybrid Acquisition Optimization: Combine DE (global) and L-BFGS-B (local)
        try:
            # Global search via Differential Evolution
            de_res = differential_evolution(acq, bounds=[(0, 1)] * self.dim, maxiter=20, popsize=10)
            # Local refinement via L-BFGS-B
            res = minimize(acq, de_res.x, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
            target_x = res.x
        except:
            target_x = np.random.rand(self.dim)
            
        return np.clip(target_x, 0, 1) * (ub - lb) + lb