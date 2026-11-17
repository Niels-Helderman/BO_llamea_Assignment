import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        
        # Initial points: Use a small Latin Hypercube approach
        n_init = max(2 * self.dim + 2, 10)
        
        for i in range(self.budget):
            if i < n_init:
                # Stratified sampling in [0, 1] mapped to bounds
                x = lb + (np.random.rand(self.dim) + np.arange(self.dim)) / (self.dim + 1) * (ub - lb)
            else:
                X_arr = (np.array(X) - lb) / (ub - lb)
                y_min = np.min(Y)
                y_mean, y_std = np.mean(Y), np.std(Y) + 1e-9
                Y_norm = (np.array(Y) - y_mean) / y_std
                
                try:
                    self.gp.fit(X_arr, Y_norm)
                    
                    def ei_acquisition(x_norm):
                        mu_norm, std_norm = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                        mu = mu_norm * y_std + y_mean
                        sigma = std_norm * y_std
                        imp = y_min - mu
                        Z = imp / (sigma + 1e-9)
                        return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z)).item()

                    best_ei = float('inf')
                    x_next = np.random.uniform(0, 1, self.dim)
                    for _ in range(8):
                        x0 = np.random.rand(self.dim)
                        res = minimize(ei_acquisition, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if res.fun < best_ei:
                            best_ei = res.fun
                            x_next = res.x
                    x = lb + x_next * (ub - lb)
                except:
                    x = np.random.uniform(lb, ub)
            
            # Ensure boundaries
            x = np.clip(x, lb, ub)
            y = float(func(x))
            
            if y < best_y:
                best_y, best_x = y, x
            
            X.append(x)
            Y.append(y)
            
        return best_y, best_x