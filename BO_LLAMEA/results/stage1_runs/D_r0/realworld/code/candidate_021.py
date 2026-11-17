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
        # Kernel with noise learning to prevent overfitting and handle objective noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5 

    def _ucb(self, x_norm, gp, beta, lb, ub):
        x = (x_norm.reshape(1, -1) * (ub - lb) + lb)
        mu, std = gp.predict(x, return_std=True)
        return -(mu - beta * std) # Minimize negative UCB

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None
        
        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # Scrambled quasi-random init
                x = lb + (ub - lb) * (np.random.rand(self.dim) * 0.95 + (i % n_init) / n_init * 0.05)
            else:
                beta = 2.0 * np.sqrt(np.log(i + 1)) # Adaptive exploration weight
                x_incumbent = self.X_hist[np.argmin(self.y_hist)]
                x_inc_norm = (x_incumbent - lb) / (ub - lb)
                
                # Trust region search to avoid global stagnation
                tr_lb = np.clip(x_inc_norm - self.tr_size, 0.0, 1.0)
                tr_ub = np.clip(x_inc_norm + self.tr_size, 0.0, 1.0)
                
                try:
                    res = minimize(
                        lambda x_n: self._ucb(x_n, self.gp, beta, lb, ub),
                        x0=np.clip(x_inc_norm, tr_lb, tr_ub),
                        bounds=list(zip(tr_lb, tr_ub)),
                        method='L-BFGS-B'
                    )
                    x = res.x * (ub - lb) + lb
                except:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            y = func(x)
            
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y, best_x = y, x
                self.tr_size = min(0.5, self.tr_size * 1.1)
            else:
                self.tr_size = max(0.05, self.tr_size * 0.9)
            
            if len(self.X_hist) >= 5:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)