import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is standard for BBOB. WhiteKernel provides numerical jitter for stability.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Latin Hypercube (LHS)
        n_init = max(2 * self.dim + 2, 10)
        X = np.array([lb + (ub - lb) * (np.random.permutation(n_init) + np.random.rand(n_init)) / (n_init + 1) for _ in range(self.dim)]).T
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def ucb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return -(mu - beta * sigma)

        for i in range(n_init, self.budget):
            # Refit GP only every few steps to save compute, except for very early stages
            if i % 3 == 0 or i < n_init + 5:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            # Adaptive beta: decay schedule to transition from exploration to exploitation
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            # Global acquisition optimization using Differential Evolution
            res = differential_evolution(
                ucb, bounds=list(zip(lb, ub)), args=(self.gp, beta),
                strategy='best1bin', popsize=10, tol=0.01, mutation=(0.5, 1), recombination=0.7, polish=False
            )
            
            next_x = res.x if res.success else np.random.uniform(lb, ub)
            
            # Execute objective with fallback
            try:
                val = func(next_x)
            except:
                val = np.inf
            
            if not np.isfinite(val):
                val = np.max(y) + 1e6
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x