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
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        n_init = max(2 * self.dim + 2, min(self.budget // 4, 12))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        
        for i in range(self.budget):
            if i < n_init:
                x_norm = sampler.random(1).flatten()
            else:
                X_norm = (np.array(X) - lb) / (ub - lb)
                y_arr = np.array(Y)
                y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-9
                Y_norm = (y_arr - y_mean) / y_std
                
                try:
                    self.gp.fit(X_norm, Y_norm)
                    def acq(x_n):
                        mu_n, std_n = self.gp.predict(x_n.reshape(1, -1), return_std=True)
                        mu = mu_n * y_std + y_mean
                        sigma = np.maximum(std_n * y_std, 1e-12)
                        imp = np.min(Y) - mu
                        return -(imp * norm.cdf(imp / sigma) + sigma * norm.pdf(imp / sigma))

                    best_val, x_next = float('inf'), np.random.rand(self.dim)
                    # Multi-start acquisition optimization
                    candidates = np.vstack([np.random.rand(15, self.dim), X_norm[np.argmin(Y_norm)] + np.random.normal(0, 0.1, (5, self.dim))])
                    for x0 in candidates:
                        res = minimize(acq, np.clip(x0, 0, 1), bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if res.fun < best_val:
                            best_val, x_next = res.fun, res.x
                    x_norm = x_next
                except:
                    x_norm = np.random.rand(self.dim)
            
            x = np.clip(lb + (ub - lb) * x_norm, lb, ub)
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x