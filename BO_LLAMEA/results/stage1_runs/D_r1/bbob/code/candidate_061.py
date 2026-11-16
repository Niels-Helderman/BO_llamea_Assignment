import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use ARD (Automatic Relevance Determination) kernel for better dimension scaling
        # and log-transforming the targets to handle large objective value variance.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling than random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, 12)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        # Track log of targets to stabilize GP training for BBOB functions
        y_log = np.log(y - np.min(y) + 1e-6)
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Periodic hyperparameter optimization
                self.gp.n_restarts_optimizer = 1 if i % 10 == 0 else 0
                self.gp.fit(X, y_log)
            except:
                pass
            
            def ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # EI on the transformed space is a heuristic for faster convergence
                z = (np.min(y_log) - mu) / sigma
                return -( (np.min(y_log) - mu) * norm.cdf(z) + sigma * norm.pdf(z) )
            
            # Multi-start acquisition optimization
            next_x = None
            best_acq = np.inf
            for _ in range(5):
                x0 = np.random.uniform(0, 1, self.dim)
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            y_log = np.log(y - np.min(y) + 1e-6)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x * (ub - lb) + lb