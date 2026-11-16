import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small white noise to handle deterministic objective jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization
        n_init = min(self.budget, max(2 * self.dim + 2, 10))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Optimization Loop
        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters (every few steps to save compute)
            if i % 3 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            # UCB Acquisition function with decaying beta (schedule: 2.0 down to 0.5)
            beta = 2.0 - 1.5 * (i / self.budget)
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Generate diverse starting points: current best, random, and center
            starts = [best_x_norm] + [np.random.rand(self.dim) for _ in range(5)]
            
            for x0 in starts:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            # Evaluate objective
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb