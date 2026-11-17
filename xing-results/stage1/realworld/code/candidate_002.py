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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0)

    def _get_ei(self, x, gp, y_min, mean, std):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        imp = y_min - mu
        Z = imp / sigma
        return (imp * norm.cdf(Z) + sigma * norm.pdf(Z)).item()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X, Y = [], []
        best_y, best_x = np.inf, None

        # Initialization: Sobol-like or Latin Hypercube would be ideal, 
        # but using uniform random for simplicity within budget constraints
        n_init = min(self.budget // 5, 2 * self.dim + 2)
        
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                # Standardize inputs to [0,1]
                X_arr = (np.array(X) - lb) / (ub - lb)
                y_mean, y_std = np.mean(Y), np.std(Y) + 1e-7
                Y_norm = (np.array(Y) - y_mean) / y_std
                
                try:
                    self.gp.fit(X_arr, Y_norm)
                    # Optimize acquisition
                    best_ei = -1
                    x_next = np.random.uniform(lb, ub)
                    for _ in range(5):
                        x0 = np.random.uniform(0, 1, self.dim)
                        res = minimize(lambda x: -self._get_ei(x, self.gp, np.min(Y_norm), y_mean, y_std),
                                       x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if -res.fun > best_ei:
                            best_ei = -res.fun
                            x_next = res.x * (ub - lb) + lb
                except:
                    x_next = np.random.uniform(lb, ub)
                x = x_next

            y = func(x)
            if y < best_y:
                best_y, best_x = y, x
            
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x