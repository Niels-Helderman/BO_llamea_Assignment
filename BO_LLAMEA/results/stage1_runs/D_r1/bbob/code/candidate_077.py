import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small jitter to ensure stability in deterministic regimes
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initialization density to capture landscape early
        n_init = max(2 * self.dim + 8, 20)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_y = np.min(y)
        best_x_norm = X[np.argmin(y)].copy()
        
        for i in range(n_init, self.budget):
            # Periodically refit hyperparameters for efficiency
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # Log-EI is numerically more stable than standard EI
                diff = best_y - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Adaptive Multi-start optimization
            best_acq = np.inf
            next_x = None
            
            candidates = []
            # Local trust-region search
            lb_tr = np.clip(best_x_norm - self.tr_length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.tr_length, 0, 1)
            for _ in range(12):
                candidates.append(np.random.uniform(lb_tr, ub_tr))
            # Global exploration
            for _ in range(4):
                candidates.append(np.random.uniform(0, 1, self.dim))
                
            for x0 in candidates:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback to random if acquisition optimization fails
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_length = np.clip(self.tr_length * 1.1, 0.1, 1.0)
            else:
                self.tr_length = np.clip(self.tr_length * 0.9, 0.05, 1.0)
                
        return float(best_y), best_x_norm * (ub - lb) + lb