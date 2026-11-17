import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Kernel with noise floor to handle numerical instability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization: Sobol Sequence for space-filling properties
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        # Iterative optimization with UCB
        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters
            self.gp.fit(X, y)
            
            # Adaptive beta: balance exploration (early) and exploitation (late)
            t = (i - n_init + 1) / (self.budget - n_init)
            beta = 2.0 * np.exp(-3.0 * t) + 0.1
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            starts = np.vstack([best_x, np.random.rand(8, self.dim)])
            
            for x0 in starts:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            # Clamp to domain
            next_x = np.clip(next_x, 0, 1)
            val = func(next_x * (ub - lb) + lb)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x * (ub - lb) + lb