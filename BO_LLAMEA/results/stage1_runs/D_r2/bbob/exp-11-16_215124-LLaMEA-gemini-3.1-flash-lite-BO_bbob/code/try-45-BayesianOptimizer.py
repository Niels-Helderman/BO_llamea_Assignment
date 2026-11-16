import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as a standard for smooth objective surfaces
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Space-filling initial design
        init_size = max(2 * self.dim + 2, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            # Only refit hyperparameters periodically to save budget/time
            if (i - init_size) % 10 == 0:
                try:
                    self.gp.fit(X, y)
                except:
                    pass
            
            def get_neg_ei(x_norm):
                x = lb + x_norm * range_val
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                imp = best_y - mu
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                return -ei

            # Hybrid acquisition optimization: Multi-start L-BFGS-B + random sampling
            best_ei = -np.inf
            candidate_norm = np.random.rand(self.dim)
            
            # Global candidates
            for _ in range(12):
                x0 = np.random.rand(self.dim)
                res = minimize(get_neg_ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    candidate_norm = res.x
            
            # Local candidates around current best to ensure exploitation
            best_x_norm = (best_x - lb) / range_val
            for _ in range(4):
                x0 = np.clip(best_x_norm + np.random.normal(0, 0.1, self.dim), 0, 1)
                res = minimize(get_neg_ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    candidate_norm = res.x

            next_x = lb + candidate_norm * range_val
            val = func(next_x)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x