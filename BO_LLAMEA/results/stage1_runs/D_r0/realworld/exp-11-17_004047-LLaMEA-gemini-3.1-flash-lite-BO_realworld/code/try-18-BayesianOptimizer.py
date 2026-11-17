import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def _ei(self, x_norm, y_best, lb, ub):
        x = (x_norm * (ub - lb) + lb).reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-12, None)
        diff = y_best - mu
        z = diff / sigma
        return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_y, best_x = float('inf'), None
        n_init = max(2 * self.dim + 2, 8)

        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                y_best = np.min(self.y_hist)
                # Multi-start acquisition optimization
                best_acq = float('inf')
                x_next = None
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(self._ei, x0, args=(y_best, lb, ub), 
                                   bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x * (ub - lb) + lb
                x = x_next if x_next is not None else lb + (ub - lb) * np.random.rand(self.dim)

            x = np.clip(x, lb, ub)
            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)

            if y < best_y:
                best_y, best_x = float(y), x

            # Efficiently fit GP
            if len(self.X_hist) >= 5:
                try:
                    # Incrementally increase alpha if fitting fails
                    self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                except:
                    self.gp.alpha += 1e-3
        
        return best_y, best_x