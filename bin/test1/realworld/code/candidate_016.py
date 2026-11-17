import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-9
                self.gp.fit(X_arr, (y_arr - y_mean) / y_std)
                x_next = self._propose_next(lb, ub, y_mean, y_std, i)
            
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            self.X.append(x_next)
            self.y.append(y_val)
            
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub, y_mean, y_std, iteration):
        # UCB with beta decreasing from 2.5 to 0.5 for exploration/exploitation balance
        beta = 0.5 + 2.0 * np.exp(-iteration / (self.budget / 3))
        
        def ucb(x):
            mu_norm, std_norm = self.gp.predict(x.reshape(1, -1), return_std=True)
            mu = mu_norm * y_std + y_mean
            std = std_norm * y_std
            return mu - beta * std

        best_x = np.random.uniform(lb, ub, self.dim)
        best_val = np.inf
        
        # Multistart optimization
        starts = np.vstack([self.X[np.argmin(self.y)], np.random.uniform(lb, ub, (8, self.dim))])
        for x0 in starts:
            res = minimize(ucb, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x