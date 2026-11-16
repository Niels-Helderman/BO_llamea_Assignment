import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a kernel with learned length scale and fixed jitter to handle deterministic nature
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8, noise_level_bounds='fixed')
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)
        self.tr_size = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial distribution
        n_init = min(self.budget, max(2 * self.dim + 2, 12))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_y = np.min(y)
        best_x_norm = X[np.argmin(y)].copy()
        
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass # Fallback to random if GP fitting fails
            
            # Log-EI Acquisition function for numerical stability
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI computation
                log_ei = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z) + np.log(1 + (sigma / diff) * (norm.pdf(z) / norm.cdf(z) + 1e-20))
                return -log_ei
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Candidates: Mix of local (Trust Region) and Global points
            candidates = [np.clip(best_x_norm + np.random.uniform(-self.tr_size, self.tr_size, self.dim), 0, 1) for _ in range(10)]
            candidates.extend([np.random.uniform(0, 1, self.dim) for _ in range(5)])
            
            for x0 in candidates:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_size = np.clip(self.tr_size * 1.1, 0.1, 0.8)
            else:
                self.tr_size = np.clip(self.tr_size * 0.9, 0.05, 0.8)
                
        return float(best_y), best_x_norm * (ub - lb) + lb