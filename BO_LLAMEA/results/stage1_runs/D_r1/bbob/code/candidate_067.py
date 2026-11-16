import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale="fixed") + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for improved space-filling initialization
        sampler = qmc.LatinHypercube(d=self.dim)
        n_init = min(self.budget, max(2 * self.dim + 2, 12))
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Multi-start acquisition optimization within a dynamic trust region
            next_x = None
            best_acq = np.inf
            for _ in range(8):
                x0 = np.clip(best_x_norm + np.random.normal(0, self.length, self.dim), 0, 1)
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive length scale logic
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = min(1.0, self.length * 1.2)
            else:
                self.length = max(0.01, self.length * 0.85)
                
        return best_y, best_x_norm * (ub - lb) + lb