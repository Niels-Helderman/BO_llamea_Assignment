import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e1)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        # Use a more stable GP: alpha handles numerical noise, n_restarts allows fitting hyperparameters
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, n_restarts_optimizer=2, normalize_y=True)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        
        # Sobol sequence for better space filling than random uniform
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = min(self.budget // 3, 2 * self.dim + 2)

        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * sampler.random(1).flatten()
            else:
                X_arr, Y_arr = np.array(X), np.array(Y)
                X_norm = (X_arr - lb) / (ub - lb)
                y_min = np.min(Y_arr)
                
                try:
                    self.gp.fit(X_norm, Y_arr)
                    def acq(x_n):
                        x_n = x_n.reshape(1, -1)
                        mu, std = self.gp.predict(x_n, return_std=True)
                        imp = y_min - mu
                        Z = imp / (std + 1e-10)
                        return -(imp * norm.cdf(Z) + std * norm.pdf(Z))

                    best_ei, x_next = -1e-9, None
                    # Multistart acquisition optimization
                    for _ in range(5):
                        x0 = np.random.rand(self.dim)
                        res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if -res.fun > best_ei:
                            best_ei, x_next = -res.fun, res.x
                    
                    if x_next is None: x_next = np.random.rand(self.dim)
                    x = lb + (ub - lb) * x_next
                except:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            # Avoid duplicate queries
            if len(X) > 0 and np.any(np.all(np.isclose(X, x), axis=1)):
                x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x