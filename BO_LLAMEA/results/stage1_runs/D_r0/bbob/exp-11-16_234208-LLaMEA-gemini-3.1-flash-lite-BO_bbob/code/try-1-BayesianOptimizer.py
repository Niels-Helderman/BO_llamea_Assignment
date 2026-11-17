import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        n_init = min(self.budget, 2 * self.dim + 1)
        
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                x = self._propose(X_obs, y_obs, lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X_obs, y_obs, lb, ub):
        X = np.array(X_obs)
        y = np.array(y_obs)
        
        # Periodic refit to manage cost
        if len(X) % 5 == 0:
            self.gp.kernel.set_params(**self.gp.kernel.get_params())
            self.gp.fit(X, y)
        else:
            self.gp.fit(X, y)
            
        y_min = np.min(y)
        
        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            with np.errstate(divide='ignore'):
                imp = y_min - mu
                z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                ei[sigma <= 0] = 0.0
            return -ei.flatten()

        # Multi-start acquisition optimization
        best_x = None
        best_acq = -np.inf
        for _ in range(8):
            x0 = np.random.uniform(lb, ub)
            res = minimize(objective, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
                
        return best_x if best_x is not None else np.random.uniform(lb, ub)