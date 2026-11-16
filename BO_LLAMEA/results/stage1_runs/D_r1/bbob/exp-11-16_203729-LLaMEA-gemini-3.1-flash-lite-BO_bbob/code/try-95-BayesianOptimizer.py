import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with fixed small jitter to prevent singular matrices
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
        
        # Initialize with Sobol sequence for better space filling than random
        n_init = min(12 + self.dim, self.budget // 3)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx]
        
        for i in range(n_init, self.budget):
            # Periodically update hyperparameters: every step until 50, then every 5 steps
            if i < 50 or (i - n_init) % 5 == 0:
                self.gp.fit(X, y)
            
            # UCB acquisition: beta decreases over time for convergence
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return -(mu - beta * sigma)
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            starts = np.random.rand(8, self.dim)
            starts = np.vstack([starts, best_x])
            
            for x0 in starts:
                try:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
                except:
                    continue
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
                
            # Perform function call
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return float(best_y), best_x * (ub - lb) + lb