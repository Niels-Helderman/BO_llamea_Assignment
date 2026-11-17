import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Using n_restarts_optimizer=0 for efficiency; fit periodically
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10)
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling coverage
        n_init = max(2, min(self.budget // 4, 10))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
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
        
        self.gp.fit(X, y)
        
        # Adaptive beta for UCB: explore more early, exploit late
        iteration = len(self.X_obs)
        beta = 2.0 * np.sqrt(np.log(iteration + 1))
        
        def objective(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            # UCB minimization: mu - beta * sigma
            return (mu - beta * sigma).flatten()[0]

        # Multi-start acquisition optimization
        best_acq = np.inf
        target_x = None
        
        # Start search from best known point and random points
        starts = [X[np.argmin(y)]] + [np.random.rand(self.dim) for _ in range(5)]
        
        for x0 in starts:
            try:
                res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    target_x = res.x
            except:
                continue
            
        if target_x is None:
            return np.random.rand(self.dim)
        return np.clip(target_x, 0, 1)