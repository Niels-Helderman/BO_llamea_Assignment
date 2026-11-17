import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        # Kernel: Matern for flexibility, WhiteKernel to handle potential noise
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design: Latin Hypercube or Sobol are preferred, but we use uniform for simplicity
        # Use a fraction of budget for exploration/initialization
        n_init = min(self.budget // 4, 10 + 2 * self.dim)
        
        for i in range(self.budget):
            if len(self.y) < n_init:
                x_next = np.random.uniform(lb, ub, self.dim)
            else:
                x_next = self._propose_next(lb, ub)
            
            y_val = func(x_next)
            self.X.append(x_next)
            self.y.append(y_val)
            
            # Update surrogate every iteration after sufficient data
            if len(self.y) >= n_init:
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-8
                y_norm = (y_arr - y_mean) / y_std
                
                # Re-fit GP every few steps or always if budget is small
                self.gp.fit(X_arr, y_norm)

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub):
        def acq(x):
            x = x.reshape(1, -1)
            mu, std = self.gp.predict(x, return_std=True)
            y_min = np.min(self.y)
            y_mean, y_std = np.mean(self.y), np.std(self.y) + 1e-8
            y_min_norm = (y_min - y_mean) / y_std
            
            z = (y_min_norm - mu) / (std + 1e-9)
            ei = (y_min_norm - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei # Minimize negative EI

        best_x = None
        best_ei = -np.inf
        # Multi-start L-BFGS-B optimization for acquisition
        for _ in range(5):
            x0 = np.random.uniform(lb, ub, self.dim)
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        return best_x