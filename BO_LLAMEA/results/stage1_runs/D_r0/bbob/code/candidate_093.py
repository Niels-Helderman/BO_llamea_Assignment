import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is standard for BBOB; WhiteKernel handles numeric stability for deterministic functions
        self.kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.iteration = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 2)
        
        X_obs = []
        y_obs = []
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_samples[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            self.iteration += 1
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        
        # Periodically re-optimize hyperparameters to save cost as data grows
        self.gp.fit(X_norm, y)
        
        # Adaptive LCB: Lower Confidence Bound. Beta decreases as we progress for convergence
        beta = 2.0 * np.log(self.iteration + 1)
        
        def lcb(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        # Multi-start acquisition optimization
        best_val = np.inf
        best_x = None
        
        # Sample points from Sobol sequence to initialize local search
        starts = qmc.Sobol(d=self.dim, scramble=True).random(10)
        
        for x0 in starts:
            try:
                res = minimize(lcb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            return np.random.uniform(lb, ub)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb