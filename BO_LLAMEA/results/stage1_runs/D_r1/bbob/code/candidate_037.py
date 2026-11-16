import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use an ensemble approach: Global GP with ARD for broad trends + Local bias
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-7, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for better initial coverage than random
        n_init = max(2 * self.dim, 10)
        X = np.array([np.random.permutation(np.linspace(0, 1, n_init)) for _ in range(self.dim)]).T
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e6)
            diff = y_min - mu
            z = diff / sigma
            return diff * norm.cdf(z) + sigma * norm.pdf(z)

        # Refit frequency: optimize hyperparameters every few steps to save compute
        for i in range(n_init, self.budget):
            try:
                # Periodic hyperparameter optimization
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Acquisition: Multi-start L-BFGS-B to maximize EI
            best_acq = -np.inf
            next_x = None
            
            # Strategy: 10 random restarts for acquisition optimization
            for _ in range(10):
                x0 = np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, 
                               bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-5)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Fallback if optimization fails or to explore
            if next_x is None or np.isnan(next_x).any():
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            
            # Update history
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
            
        return best_y, best_x_norm * (ub - lb) + lb