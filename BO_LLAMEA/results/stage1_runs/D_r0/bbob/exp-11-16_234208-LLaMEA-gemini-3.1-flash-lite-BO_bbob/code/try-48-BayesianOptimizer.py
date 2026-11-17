import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import differential_evolution

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Using a fixed internal optimizer to avoid expensive re-fitting if data is stable
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10)
        self.tr_length = 0.5 

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
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.gp.fit(X_norm, y)
        y_min = np.min(y)
        
        def acquisition(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()[0]

        # Use Differential Evolution for global acquisition maximization
        try:
            res = differential_evolution(
                acquisition, 
                bounds=[(0, 1)] * self.dim, 
                strategy='best1bin', 
                maxiter=50, 
                popsize=10, 
                tol=0.01,
                polish=True
            )
            return np.clip(res.x, 0, 1) * (ub - lb) + lb
        except:
            # Fallback to random point
            return np.random.uniform(lb, ub)