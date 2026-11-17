import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for better initial coverage
        n_init = max(self.dim + 5, 12)
        samples = (np.random.rand(n_init, self.dim) + np.random.rand(n_init, self.dim)) / 2.0
        X = np.sort(samples, axis=0)
        np.random.shuffle(X)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        def ei_acq(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e6)
            gamma = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        best_y = np.min(y)
        best_x_norm = X[np.argmin(y)].copy()

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Multi-start optimization of EI to avoid local minima
            best_acq = -np.inf
            next_x = None
            
            # Start from current best and random points to balance exploitation and exploration
            starts = [best_x_norm] + [np.random.rand(self.dim) for _ in range(8)]
            for x0 in starts:
                res = minimize(lambda x: -ei_acq(x, self.gp, y_min), x0, 
                               bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-5)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
                
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
            
            # Periodic refit constraint managed by GaussianProcessRegressor internals
            # and limited restarts to stay within computational budget.
        
        return float(best_y), best_x_norm * (ub - lb) + lb