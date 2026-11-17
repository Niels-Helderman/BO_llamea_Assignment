import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=2)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        n_init = max(2 * self.dim + 2, 10)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        # 2. Sequential Optimization
        for i in range(n_init, self.budget):
            # Periodically refit GP hyperparameters
            self.gp.fit(X, y)
            
            # Beta schedule for UCB (Exploration -> Exploitation)
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            def ucb_acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Multi-start acquisition optimization
            best_acq_val = np.inf
            next_x = np.random.uniform(0, 1, self.dim)
            
            for _ in range(5):
                x0 = np.random.uniform(0, 1, self.dim)
                res = minimize(ucb_acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    next_x = res.x
            
            # Evaluate objective
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x
        
        return best_y, best_x * (ub - lb) + lb