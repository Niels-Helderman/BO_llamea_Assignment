import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel with fixed noise floor for numerical robustness
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization: Sobol sequence for better space filling
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Adaptive beta schedule for UCB: starts high for exploration, decreases for exploitation
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                beta = 2.0 * np.sqrt(np.log(i + 1)) # Standard UCB exploration schedule
            except:
                beta = 0.1 # Fallback to exploitation if GP fitting fails
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma # LCB (Minimization)

            # Optimization of acquisition function via multi-start L-BFGS-B
            best_acq = np.inf
            next_x = None
            
            # Use points around current best + random samples for robust global coverage
            candidates = [best_x_norm]
            candidates.extend(np.random.rand(8, self.dim))
            
            for x0 in candidates:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback for numerical instability
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb