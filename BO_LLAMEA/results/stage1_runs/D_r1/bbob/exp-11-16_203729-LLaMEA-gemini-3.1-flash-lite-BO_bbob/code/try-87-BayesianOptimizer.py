import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use WhiteKernel for numerical stability with deterministic functions
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
        
        # Sobol sequence for better space filling
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically re-optimize hyperparameters to balance cost/performance
            if (i - n_init) % 5 == 0:
                self.gp.n_restarts_optimizer = 1
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X, y)
            except:
                # Fallback to random if GP fails
                next_x = np.random.rand(self.dim)
            else:
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    # Use Log-EI to handle numerical stability
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start L-BFGS-B optimization
                best_acq = np.inf
                next_x = np.random.rand(self.dim)
                for start_x in [best_x_norm] + [np.random.rand(self.dim) for _ in range(4)]:
                    res = minimize(acq, start_x, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            
            # Clamp next_x to [0,1]
            next_x = np.clip(next_x, 0, 1)
            
            # Check for duplicate candidates to avoid redundant evaluations
            if np.any(np.all(np.abs(X - next_x) < 1e-5, axis=1)):
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb