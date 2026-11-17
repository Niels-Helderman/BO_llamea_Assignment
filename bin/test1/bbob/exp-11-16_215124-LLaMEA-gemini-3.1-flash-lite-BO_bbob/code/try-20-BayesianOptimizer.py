import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is well-suited for physical black-box functions
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # 1. Sobol Initialization for better space filling than random
        init_size = max(2 * self.dim + 4, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(init_size)
        X_init = lb + sample * scale
        
        X, y = [], []
        for x in X_init:
            X.append(x)
            y.append(func(x))
        
        X, y = np.array(X), np.array(y)
        
        for i in range(init_size, self.budget):
            # 2. GP Refitting: periodically optimize hyperparameters
            if i % 15 == 0:
                self.gp.set_params(n_restarts_optimizer=2)
            else:
                self.gp.set_params(n_restarts_optimizer=0)
            
            self.gp.fit(X, y)
            
            # 3. UCB Acquisition with adaptive beta (log-schedule)
            # Beta decays to allow transition from exploration to exploitation
            beta = 2.0 * np.log(i + 1)
            
            def ucb(x_norm):
                x = lb + x_norm * scale
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return -(mu - beta * sigma)
            
            # 4. Multi-start acquisition optimization
            best_val = np.inf
            next_x = None
            for _ in range(8):
                x0 = np.random.rand(self.dim)
                res = minimize(ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    next_x = lb + res.x * scale
            
            # Fallback for numerical instability
            if next_x is None:
                next_x = lb + np.random.rand(self.dim) * scale
                
            y_next = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, y_next)
            
        best_idx = np.argmin(y)
        return float(y[best_idx]), X[best_idx]