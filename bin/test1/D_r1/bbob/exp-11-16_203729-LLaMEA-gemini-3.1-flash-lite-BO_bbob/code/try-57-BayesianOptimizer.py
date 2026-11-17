import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-10)
        # Use alpha for numerical stability; avoid frequent kernel re-optimization
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling is often more robust for small budgets than Sobol
        sampler = qmc.LatinHypercube(d=self.dim)
        n_init = min(self.budget, 2 * self.dim + 2)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Refit GP only when data is sufficient and meaningful
            if i % 3 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Search for next point in trust region
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            best_acq = np.inf
            next_x = None
            # Multi-start optimization
            for _ in range(5):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update best and adjust trust region
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = min(self.length * 1.2, 1.0)
            else:
                self.length = max(self.length * 0.8, 0.05)
                
        return best_y, best_x_norm * (ub - lb) + lb