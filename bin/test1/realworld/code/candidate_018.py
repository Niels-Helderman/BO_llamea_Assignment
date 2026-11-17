import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=1, normalize_y=False)

    def _lhs(self, n, dim):
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = (np.random.permutation(n) + np.random.rand(n)) / n
        return samples

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        n_init = max(2 * self.dim + 1, min(self.budget // 4, 15))
        
        init_samples = self._lhs(n_init, self.dim)

        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * init_samples[i]
            else:
                X_norm = (np.array(X) - lb) / (ub - lb)
                y_arr = np.array(Y)
                y_min = np.min(y_arr)
                y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-9
                Y_norm = (y_arr - y_mean) / y_std
                
                try:
                    self.gp.fit(X_norm, Y_norm)
                    def acq(x_n):
                        mu_n, std_n = self.gp.predict(x_n.reshape(1, -1), return_std=True)
                        mu = mu_n * y_std + y_mean
                        sigma = np.clip(std_n * y_std, 1e-12, None)
                        Z = (y_min - mu) / sigma
                        return -( (y_min - mu) * norm.cdf(Z) + sigma * norm.pdf(Z) )

                    best_ei, x_next = -1.0, np.random.rand(self.dim)
                    for x0 in np.vstack([np.random.rand(8, self.dim), np.random.rand(self.dim)]):
                        res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                        if -res.fun > best_ei:
                            best_ei, x_next = -res.fun, res.x
                    x = lb + (ub - lb) * x_next
                except:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x