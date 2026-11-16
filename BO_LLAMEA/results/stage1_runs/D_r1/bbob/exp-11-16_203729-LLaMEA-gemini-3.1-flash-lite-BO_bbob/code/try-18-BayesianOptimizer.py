import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 kernel with WhiteKernel to account for potential objective noise/jitter
        self.kernel = Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial exploration: Latin Hypercube sampling approximation
        n_init = max(2 * self.dim + 2, 8)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Iterative optimization with UCB
        for i in range(n_init, self.budget):
            # Periodically re-fit GP
            if i % 2 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            # Adaptive beta schedule for UCB
            # Beta decays from 2.0 to 0.5 to transition from exploration to exploitation
            beta = 2.0 * (1.0 - (i / self.budget)) + 0.5
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Optimize acquisition with multi-start
            best_acq = np.inf
            next_x = np.random.uniform(0, 1, self.dim)
            
            for _ in range(6):
                x0 = np.random.uniform(0, 1, self.dim)
                res = minimize(ucb, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Evaluate function
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                
        return best_y, best_x_norm * (ub - lb) + lb