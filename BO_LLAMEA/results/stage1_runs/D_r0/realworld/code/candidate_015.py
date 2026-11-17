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
        # Robust kernel setup with noise handling
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=1)
        self.tr_size = 0.5

    def _ucb(self, x_norm, gp, beta, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, sigma = gp.predict(x, return_std=True)
        return (mu - beta * sigma).item()

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # Latin Hypercube/Sobol-lite sampling
                x = lb + (ub - lb) * ((np.arange(self.dim) + i) % n_init / n_init)
            else:
                # Update model
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                
                # UCB with beta decay (Exploration to Exploitation)
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                x_inc = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_inc - lb) / (ub - lb)
                
                # Local search with UCB
                tr_lb = np.maximum(0, x_inc_norm - self.tr_size)
                tr_ub = np.minimum(1, x_inc_norm + self.tr_size)
                
                res = minimize(
                    lambda x_n: self._ucb(x_n, self.gp, beta, lb, ub),
                    x0=x_inc_norm,
                    bounds=list(zip(tr_lb, tr_ub)),
                    method='L-BFGS-B'
                )
                x = res.x * (ub - lb) + lb
                
                # Adaptive TR update
                if res.fun < np.min(self.y_hist):
                    self.tr_size = min(0.8, self.tr_size * 1.1)
                else:
                    self.tr_size = max(0.05, self.tr_size * 0.8)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                
        return float(best_y), np.array(best_x)