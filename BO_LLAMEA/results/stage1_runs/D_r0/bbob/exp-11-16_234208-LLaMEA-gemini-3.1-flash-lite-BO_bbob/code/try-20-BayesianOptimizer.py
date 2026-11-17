import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small noise term for deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for space-filling initialization
        n_init = max(self.dim + 1, min(self.budget // 4, 10))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = init_points[i]
                x = x_norm * (ub - lb) + lb
            else:
                # Refit GP only when needed
                self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                # Adaptive UCB parameter: decreases over time to converge on exploitation
                t = i / self.budget
                beta = 2.0 * (1.0 - t) + 0.1
                x = self._propose_ucb(beta, lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose_ucb(self, beta, lb, ub):
        def objective(x_norm):
            x_norm = x_norm.reshape(1, -1)
            mu, sigma = self.gp.predict(x_norm, return_std=True)
            # UCB for minimization: mu - beta * sigma
            return mu - beta * sigma

        best_val = np.inf
        best_x = None
        
        # Optimize acquisition using multi-start from best observed point and random points
        starts = [self.gp.X_train_[np.argmin(self.gp.y_train_)]]
        for _ in range(5):
            starts.append(np.random.rand(self.dim))
            
        for x0 in starts:
            try:
                res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            best_x = np.random.rand(self.dim)
            
        return np.clip(best_x, 0, 1) * (ub - lb) + lb