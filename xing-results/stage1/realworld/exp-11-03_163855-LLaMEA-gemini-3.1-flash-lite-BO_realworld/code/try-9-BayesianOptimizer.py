import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=3, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, min(self.budget // 4, 15))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose_next(lb, ub)
            
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            self.X.append(x_next)
            self.y.append(y_val)
            
            if len(self.y) >= n_init:
                try:
                    self.gp.fit(np.array(self.X), np.array(self.y))
                except:
                    pass

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub):
        y_min = np.min(self.y)
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            std = np.maximum(std, 1e-9)
            z = (y_min - mu) / std
            ei = (y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei[0]

        best_x = np.random.uniform(lb, ub, self.dim)
        best_ei = -np.inf
        
        # Multi-start optimization with local search and random samples
        for x0 in [np.random.uniform(lb, ub, self.dim) for _ in range(8)]:
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
                
        return best_x