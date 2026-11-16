import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=True)

    def _get_ei(self, x, gp, y_best):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        diff = y_best - mu
        from scipy.stats import norm
        z = diff / sigma
        ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei.item()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X, y = [], []
        best_y, best_x = np.inf, None

        # Initialization with Latin Hypercube-like spread
        init_size = min(self.budget // 4, 2 * self.dim + 2)
        for i in range(init_size):
            x = np.random.uniform(lb, ub)
            f = func(x)
            X.append(x); y.append(f)
            if f < best_y: best_y, best_x = f, x

        # Bayesian Optimization Loop
        for i in range(init_size, self.budget):
            X_arr, y_arr = np.array(X), np.array(y)
            
            # Periodically optimize hyperparameters
            optimizer = "fmin_l_bfgs_b" if i % 10 == 0 else "none"
            self.gp.optimizer = optimizer
            self.gp.fit(X_arr, y_arr)

            # Maximize EI using multi-start
            best_ei = -np.inf
            next_x = np.random.uniform(lb, ub)
            for _ in range(5):
                start_x = np.random.uniform(lb, ub)
                res = minimize(self._get_ei, start_x, args=(self.gp, np.min(y_arr)), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    next_x = res.x
            
            f = func(next_x)
            X.append(next_x); y.append(f)
            if f < best_y: best_y, best_x = f, next_x
            
        return float(best_y), best_x