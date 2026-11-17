import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Constant kernel scale + Matern(nu=2.5) for smooth surfaces; jitter in alpha handles deterministic nature
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=2
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        # Latin Hypercube Sampling for robust initial space filling
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_init)
        init_points = qmc.scale(sample, lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i]
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            # Ensure x is within bounds even if optimization drifts
            x = np.clip(x, lb, ub)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        # Refit hyperparameters every iteration; GPR is fast enough for 600 calls with this budget
        try:
            self.gp.fit(X, y)
        except:
            return np.random.uniform(lb, ub)
            
        y_min = np.min(y)
        
        def objective(x):
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.flatten()

        best_x = None
        best_acq = -np.inf
        
        # Multi-start optimization of EI
        starts = np.random.uniform(lb, ub, (10, self.dim))
        for x0 in starts:
            res = minimize(objective, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
                
        return best_x if best_x is not None else np.random.uniform(lb, ub)