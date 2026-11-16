import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 2.5 is a robust default for unknown black-box functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial space-filling using Latin Hypercube
        init_size = max(2 * self.dim + 2, 12)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(y)].copy()
        best_y = float(np.min(y))
        
        for i in range(init_size, self.budget):
            try:
                # Refit kernel hyperparams every 20 iterations
                self.gp.n_restarts_optimizer = 2 if (i % 20 == 0) else 0
                self.gp.fit(X, y)
                
                # Adaptive UCB beta: logarithmic decay to encourage exploitation late-game
                beta = 2.0 * np.log(i + 1)
                
                def get_ucb(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, std = self.gp.predict(x, return_std=True)
                    return mu - beta * std  # Minimizing negative UCB
                
                # Optimize acquisition using multiple random starts
                best_acq = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val
            
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x