import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel setup: Matern 5/2 is well-suited for BBOB smooth functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial points: Latin Hypercube Sampling is more robust for 6D than pure Sobol
        n_init = min(self.budget, 2 * self.dim + 6)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Fit GP with numerical stability
            try:
                self.gp.fit(X, y)
            except:
                pass # Fallback to existing GP if fit fails
                
            # Log-Expected Improvement for better numerical range
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                # Log-EI calculation
                f = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -f if f > 0 else 0

            # Multi-start acquisition optimization
            best_acq_val = -np.inf
            next_x = np.random.rand(self.dim)
            
            # Combine random sampling with local optimization
            starts = np.random.rand(8, self.dim)
            # Add the current best point to the starting set
            starts[0] = best_x_norm
            
            for x0 in starts:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                if -res.fun > best_acq_val:
                    best_acq_val = -res.fun
                    next_x = res.x
            
            # Evaluate objective
            val = func(next_x * (ub - lb) + lb)
            
            # Update data
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb