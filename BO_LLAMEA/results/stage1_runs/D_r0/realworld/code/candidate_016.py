import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # Robust kernel: Matern 5/2 is standard for black-box; WhiteKernel handles noise
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)

    def _get_ucb(self, x_norm, beta):
        x = x_norm.reshape(1, -1)
        mu, std = self.gp.predict(x, return_std=True)
        return mu - beta * std

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = float('inf')
        best_x = None
        
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # Deterministic-like spread using golden ratio sequence for initial sampling
                phi = (1 + 5**0.5) / 2
                x = lb + (ub - lb) * (np.array([(i * phi**j) % 1.0 for j in range(self.dim)]))
            else:
                # Update GP model
                try:
                    X_scaled = (np.array(self.X_hist) - lb) / (ub - lb)
                    self.gp.fit(X_scaled, np.array(self.y_hist))
                except:
                    pass
                
                # Adaptive UCB beta: logarithmic decay to favor exploitation later
                beta = 2.0 * np.log(i + 1)
                
                # Global acquisition optimization with multi-start
                best_acq = float('inf')
                x_next = np.random.rand(self.dim)
                
                for _ in range(5):
                    x_start = np.random.rand(self.dim)
                    res = minimize(lambda x_n: self._get_ucb(x_n, beta), x_start, 
                                   bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
                
                x = x_next * (ub - lb) + lb
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                    
        return float(best_y), np.array(best_x)