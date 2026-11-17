import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Initialize GP with Matern 5/2, constant kernel for scale, and fixed small jitter
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=True)
        self.min_y = np.inf
        self.best_x = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X, y = [], []
        
        # Initial sampling: Latin Hypercube-like via Sobol or random for diversity
        n_init = min(10, self.budget // 4)
        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                # Acquisition function: Expected Improvement
                y_min = np.min(y)
                def acq(x_scaled):
                    x_raw = lb + x_scaled * (ub - lb)
                    mu, sigma = self.gp.predict(x_raw.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = y_min - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start acquisition optimization
                best_acq = np.inf
                x_best_acq = None
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_best_acq = res.fun, res.x
                x = lb + x_best_acq * (ub - lb)

            f = float(func(x))
            if f < self.min_y:
                self.min_y, self.best_x = f, x
            
            X.append(x)
            y.append(f)
            
            # Periodically refit hyperparameters
            if len(X) > 5:
                if len(X) < 50 or len(X) % 10 == 0:
                    self.gp.fit(np.array(X), np.array(y))
        
        return self.min_y, self.best_x