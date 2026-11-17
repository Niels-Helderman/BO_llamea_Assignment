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
        # Use a flexible kernel with white noise to handle potential objective noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=1)

    def _get_ucb(self, x, gp, beta):
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        return mu - beta * sigma

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)

        n_init = max(2 * self.dim + 2, 8)
        
        for i in range(self.budget):
            if i < n_init:
                # LHS-like stratified sampling
                x = lb + (ub - lb) * (np.random.rand(self.dim) + (i % n_init)) / n_init
            else:
                # Adaptive beta schedule for UCB exploration-exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                # Global acquisition optimization with fallback
                best_ucb = np.inf
                x_next = lb + (ub - lb) * np.random.rand(self.dim)
                
                # Multi-start optimization
                for _ in range(3):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(lambda x: self._get_ucb(x, self.gp, beta), x0=x0, 
                                   bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_ucb:
                        best_ucb = res.fun
                        x_next = res.x
            
            x = np.clip(x, lb, ub)
            
            # Distance-based duplicate rejection
            if len(self.X_hist) > 0:
                dist = np.min(np.linalg.norm(np.array(self.X_hist) - x, axis=1))
                if dist < 1e-5:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            # Retrain GP periodically
            if len(self.X_hist) >= 5:
                try:
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    pass
                    
        return float(best_y), np.array(best_x)