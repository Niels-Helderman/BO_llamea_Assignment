import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = np.inf, None

        # Initialization: Use a slightly higher initial point count for 6D space
        n_init = max(2 * self.dim + 2, 10)
        
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                X_scaled = (np.array(X) - lb) / (ub - lb)
                y_min = np.min(Y)
                y_mean, y_std = np.mean(Y), np.std(Y) + 1e-8
                Y_norm = (np.array(Y) - y_mean) / y_std
                
                try:
                    self.gp.fit(X_scaled, Y_norm)
                    
                    def acquisition(x_norm):
                        mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                        mu = mu * y_std + y_mean
                        sigma = sigma * y_std
                        imp = y_min - mu
                        Z = imp / (sigma + 1e-10)
                        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                        return -ei.item()

                    best_ei = -np.inf
                    x_next = np.random.uniform(lb, ub)
                    # Multistart acquisition optimization
                    for _ in range(8):
                        x0 = np.random.uniform(0, 1, self.dim)
                        res = minimize(acquisition, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if -res.fun > best_ei:
                            best_ei = -res.fun
                            x_next = res.x * (ub - lb) + lb
                except:
                    x_next = np.random.uniform(lb, ub)
                x = np.clip(x_next, lb, ub)

            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x