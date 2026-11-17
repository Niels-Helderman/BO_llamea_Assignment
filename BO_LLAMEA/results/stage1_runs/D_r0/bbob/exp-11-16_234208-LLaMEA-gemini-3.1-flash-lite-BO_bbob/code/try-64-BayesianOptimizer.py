import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness and white kernel for numerical stability
        self.kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=2)
        self.n_init = min(budget // 3, 2 * dim + 1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X, y = [], []
        
        # Sobol sequence for better space-filling than random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        samples = sampler.random(self.budget)
        
        for i in range(self.budget):
            if i < self.n_init:
                x = samples[i] * (ub - lb) + lb
            else:
                x = self._acquire(np.array(X), np.array(y), lb, ub)
            
            val = float(func(x))
            X.append(x)
            y.append(val)
            
        best_idx = np.argmin(y)
        return float(y[best_idx]), np.array(X[best_idx])

    def _acquire(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y_min = np.min(y)
        try:
            self.gp.fit(X_norm, y)
            
            def ei(x_n):
                mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                from scipy.stats import norm
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            best_x = None
            best_val = 0
            # Multi-start optimization
            for x0 in np.vstack([X_norm[np.argmin(y)], np.random.rand(5, self.dim)]):
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-5)
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x
            
            if best_x is not None:
                return np.clip(best_x, 0, 1) * (ub - lb) + lb
        except:
            pass
        
        # Fallback: Random search
        return np.random.uniform(lb, ub)