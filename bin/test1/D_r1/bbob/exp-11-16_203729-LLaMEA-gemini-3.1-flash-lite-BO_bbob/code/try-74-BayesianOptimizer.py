import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is well-suited for BBOB functions
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial exploration
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Iterative optimization with UCB
        for i in range(n_init, self.budget):
            # Periodically update kernel hyperparameters to save cost
            if i % 4 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            # Adaptive UCB beta: decaying schedule for convergence
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            def acq(x):
                mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                # Lower Confidence Bound (Minimization)
                return mu - beta * std

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Diverse starting points for optimization
            starts = np.vstack([best_x_norm, np.random.rand(5, self.dim)])
            for x0 in starts:
                res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            # Execute function
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb