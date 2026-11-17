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
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        best_y, best_x = float('inf'), None
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose(lb, ub, i)
            
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            
            self.X.append(x_next)
            self.y.append(y_val)
            
            if y_val < best_y:
                best_y, best_x = y_val, x_next
                
            if len(self.y) >= n_init and (i % 2 == 0 or i < n_init + 5):
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                y_std = np.std(y_arr) + 1e-7
                self.gp.fit(X_arr, (y_arr - np.mean(y_arr)) / y_std)
                
        return best_y, best_x

    def _propose(self, lb, ub, iteration):
        # UCB acquisition with decreasing exploration factor
        beta = 2.0 * np.log(iteration + 1)
        y_mean, y_std = np.mean(self.y), np.std(self.y) + 1e-7
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return (mu * y_std + y_mean) - np.sqrt(beta) * (std * y_std)

        best_x = np.random.uniform(lb, ub, self.dim)
        min_acq = np.inf
        
        # Multi-start optimization
        for _ in range(5):
            x0 = np.random.uniform(lb, ub, self.dim)
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < min_acq:
                min_acq, best_x = res.fun, res.x
        return best_x