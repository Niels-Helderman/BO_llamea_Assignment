import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for robust initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, 10)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx]
        
        for i in range(n_init, self.budget):
            # Refit GP with periodic hyperparameter optimization
            self.gp.n_restarts_optimizer = 1 if i % 10 == 0 else 0
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Adaptive beta schedule for UCB (cools from 3.0 to 0.5)
            beta = 3.0 * (1.0 - (i / self.budget)) + 0.5
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Multi-start optimization of acquisition function
            best_acq = np.inf
            next_x = np.random.rand(self.dim)
            
            for _ in range(5):
                x0 = np.random.rand(self.dim)
                res = minimize(ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Evaluate objective
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x * (ub - lb) + lb