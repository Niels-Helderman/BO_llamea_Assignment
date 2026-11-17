import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        n_init = max(2, min(self.budget // 4, 2 * self.dim + 1))

        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.uniform(0, 1, self.dim)
            else:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                X_norm = (X_arr - lb) / (ub - lb)
                y_min = np.min(Y_arr)
                y_mean, y_std = np.mean(Y_arr), np.std(Y_arr) + 1e-9
                Y_norm = (Y_arr - y_mean) / y_std
                
                try:
                    self.gp.fit(X_norm, Y_norm)
                    def acq(x_n):
                        mu_n, std_n = self.gp.predict(x_n.reshape(1, -1), return_std=True)
                        mu = mu_n * y_std + y_mean
                        sigma = np.maximum(std_n * y_std, 1e-12)
                        imp = y_min - mu
                        Z = imp / sigma
                        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                        return -ei

                    best_ei, x_next = -1.0, lb + (ub - lb) * np.random.uniform(0, 1, self.dim)
                    for _ in range(8):
                        x0 = np.random.uniform(0, 1, self.dim)
                        res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if -res.fun > best_ei:
                            best_ei, x_next = -res.fun, res.x * (ub - lb) + lb
                except:
                    x_next = lb + (ub - lb) * np.random.uniform(0, 1, self.dim)
                x = np.clip(x_next, lb, ub)
            
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x