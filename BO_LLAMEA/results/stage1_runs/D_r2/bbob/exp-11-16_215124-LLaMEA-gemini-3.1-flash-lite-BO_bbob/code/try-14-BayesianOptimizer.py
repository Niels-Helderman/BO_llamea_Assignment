import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is well-suited for BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Latin Hypercube-like initialization for better coverage
        init_size = max(2 * self.dim + 2, 10)
        X, y = [], []
        for i in range(init_size):
            x = lb + (np.random.rand(self.dim) + i) / init_size * range_val
            X.append(x)
            y.append(func(x))
        
        X, y = np.array(X), np.array(y)
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        for i in range(init_size, self.budget):
            try:
                # Update model
                self.gp.fit(X, y)
                
                # Adaptive Beta schedule for UCB: explore early, exploit late
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                def get_ucb(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    return mu - beta * sigma # Minimization: minimize lower bound

                # Multi-start optimization of acquisition function
                best_val = np.inf
                next_x = None
                for _ in range(8):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x = lb + res.x * range_val
            except:
                next_x = np.random.uniform(lb, ub)

            # Observation
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
            
            # Periodically rebuild GP to avoid memory/complexity issues if needed, 
            # though scikit-learn fits are efficient for this budget.
            
        return float(best_y), best_x