import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )
        self.init_size = min(budget // 4, 2 * dim + 5)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs, y_obs = [], []
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(self.init_size)
        
        for i in range(self.budget):
            if i < self.init_size:
                x_norm = init_points[i]
            else:
                X_norm = (np.array(X_obs) - lb) / (ub - lb)
                y = np.array(y_obs)
                
                # Periodically refit hyperparameters for efficiency
                self.gp.kernel.set_params(**self.kernel.get_params())
                self.gp.fit(X_norm, y)
                
                # Adaptive UCB: Beta schedule
                beta = 2.0 * np.log(i + 1)
                
                def ucb(x_n):
                    x_n = x_n.reshape(1, -1)
                    mu, std = self.gp.predict(x_n, return_std=True)
                    return mu - beta * std
                
                x_norm = self._optimize_acq(ucb, X_norm, y)
                
            x = x_norm * (ub - lb) + lb
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _optimize_acq(self, acq_func, X_norm, y):
        best_val = np.inf
        best_x = np.random.rand(self.dim)
        
        # Multi-start optimization
        starts = np.vstack([X_norm[np.argmin(y)], np.random.rand(8, self.dim)])
        for x0 in starts:
            res = minimize(acq_func, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x