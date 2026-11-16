import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use ARD (Automatic Relevance Determination) with anisotropic length scales
        # Matern 5/2 is well-suited for BBOB, log-transforming y helps with skewed objectives
        kernel = Matern(nu=2.5, length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for robust initial space-filling
        sampler = qmc.LatinHypercube(d=self.dim, scramble=True)
        n_init = min(self.budget // 5, 20)
        X_norm = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_norm])
        
        best_y = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        for i in range(n_init, self.budget):
            # Transform targets to log-space to handle high dynamic range often found in BBOB
            y_log = np.log(y - np.min(y) + 1e-6)
            
            # Refit periodically
            optimizer = "fmin_l_bfgs_b" if i % 10 == 0 else None
            self.gp.kernel.set_params(whitekernel__noise_level=1e-8)
            self.gp.fit(X_norm, y_log)
            
            def ei(x):
                mu_log, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                mu = np.exp(mu_log)
                sigma = np.maximum(sigma, 1e-12)
                # Use current log-incumbent for EI target
                target = np.min(y_log)
                z = (target - mu_log) / sigma
                return -( (target - mu_log) * norm.cdf(z) + sigma * norm.pdf(z) )
            
            # Multi-start acquisition optimization
            candidates = []
            for _ in range(8):
                x0 = np.random.rand(self.dim)
                res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                candidates.append((res.fun, res.x))
            
            next_x = min(candidates, key=lambda x: x[0])[1]
            
            # Black-box query
            val = func(next_x * (ub - lb) + lb)
            
            X_norm = np.vstack([X_norm, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = val
                best_x_norm = next_x
                
        return float(best_y), best_x_norm * (ub - lb) + lb