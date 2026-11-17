import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Using a slightly higher alpha for numerical stability in constrained spaces
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5
        self.min_tr_size = 0.05
        self.max_tr_size = 1.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization
        n_init = max(2 * self.dim + 4, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Only refit hyperparameters every few steps to save compute
            if (i - n_init) % 5 == 0:
                self.gp.kernel.set_params(**self.kernel.get_params())
                self.gp.fit(X, y)
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))

            # Adaptive Trust-Region acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Local candidates (Trust Region) + Global fallback
            candidates = []
            lb_tr = np.clip(best_x_norm - self.tr_size, 0, 1)
            ub_tr = np.clip(best_x_norm + self.tr_size, 0, 1)
            for _ in range(10):
                candidates.append(np.random.uniform(lb_tr, ub_tr))
            for _ in range(3):
                candidates.append(np.random.uniform(0, 1, self.dim))
                
            for x0 in candidates:
                try:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
                except: continue
            
            next_x = next_x if next_x is not None else np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_size = min(self.tr_size * 1.2, self.max_tr_size)
            else:
                self.tr_size *= 0.8
                if self.tr_size < self.min_tr_size:
                    self.tr_size = 0.5 # Reset TR if it shrinks too much
                
        return best_y, best_x_norm * (ub - lb) + lb