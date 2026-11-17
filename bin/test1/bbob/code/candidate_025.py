import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        init_size = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * scale
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        def get_ei(x_norm):
            x = lb + x_norm * scale
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            Z = (best_y - mu) / sigma
            return -( (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z) )

        for i in range(init_size, self.budget):
            try:
                self.gp.fit(X, y)
                best_ei = -np.inf
                next_x_norm = None
                
                # Multi-start optimization of EI
                starts = np.vstack([
                    (best_x - lb) / scale,
                    np.random.rand(8, self.dim)
                ])
                
                for x0 in starts:
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * scale
            except:
                next_x = lb + np.random.rand(self.dim) * scale

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
            
            # Periodically re-optimize kernel hyperparameters less frequently to save budget/time
            if i % 50 == 0:
                self.gp.kernel_ = self.gp.kernel
                self.gp.fit(X, y)
                
        return float(best_y), best_x