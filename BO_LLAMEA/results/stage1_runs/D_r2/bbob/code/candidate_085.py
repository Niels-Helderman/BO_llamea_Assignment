import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Initial exploration: LHS-Sobol sequence
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(init_size)
        X = lb + X_norm * scale
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            # Hyperparameter update policy
            if (i - init_size) % 15 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            self.gp.fit(X, y)
            
            # Acquisition: Expected Improvement
            def ei(x_n):
                x = lb + x_n * scale
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Acquisition optimization: Multi-start L-BFGS-B
            best_ei = -np.inf
            candidate = np.random.rand(self.dim)
            
            for _ in range(10):
                x0 = np.random.rand(self.dim)
                res = minimize(ei, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    candidate = res.x
            
            next_x = lb + candidate * scale
            
            # Safety: ensure finite, unique points
            if np.any(np.all(np.isclose(X, next_x), axis=1)):
                next_x = lb + np.random.rand(self.dim) * scale
                
            val = func(next_x)
            
            # Update history
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x