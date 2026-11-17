import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        # Use Matern 5/2 as it is well-suited for physical black-box functions
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol sequence for better space coverage
        n_init = max(2 * self.dim, 10)
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
            
            # Refit GP model
            X_arr = np.array(self.X)
            y_arr = np.array(self.y)
            y_min = np.min(y_arr)
            y_std = np.std(y_arr) + 1e-9
            self.gp.fit(X_arr, (y_arr - y_min) / y_std)

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub):
        y_min = np.min(self.y)
        y_std = np.std(self.y) + 1e-9
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            # Standardized EI calculation
            imp = (y_min - (mu * y_std + y_min))
            z = imp / (std * y_std + 1e-12)
            ei = imp * norm.cdf(z) + (std * y_std) * norm.pdf(z)
            return -ei[0]

        best_x = np.random.uniform(lb, ub, self.dim)
        best_val = np.inf
        
        # Multi-start L-BFGS-B optimization of the acquisition function
        for i in range(8):
            x0 = np.random.uniform(lb, ub, self.dim) if i > 0 else self.X[np.argmin(self.y)]
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x