import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel for capturing feature-specific sensitivity
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.rf = RandomForestRegressor(n_estimators=50, min_samples_split=2)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically update hyperparams; use GP for exploitation and RF for global uncertainty
            self.gp.fit(X, y)
            self.rf.fit(X, y)
            
            def acq(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                # Ensemble-weighted EI: utilize RF variance to encourage exploration in poorly sampled areas
                rf_pred = self.rf.predict(x)
                sigma = np.sqrt(np.maximum(sigma**2 + np.var([est.predict(x) for est in self.rf.estimators_]), 1e-12))
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))

            best_acq = np.inf
            next_x = None
            
            # Hybrid search: Trust region local search + Global random candidates
            candidates = [best_x_norm + np.random.normal(0, self.length / 3, self.dim) for _ in range(8)]
            candidates += [np.random.rand(self.dim) for _ in range(4)]
            
            for x0 in candidates:
                x0 = np.clip(x0, 0, 1)
                res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            next_x = np.clip(next_x, 0, 1)
            val = func(next_x * (ub - lb) + lb)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.9, 0.02, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb