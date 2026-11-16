import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel setup with fixed noise jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial exploration with Sobol sequence
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters only when necessary
            try:
                self.gp.fit(X, y)
            except:
                pass # Continue with previous GP fit if update fails
            
            # Adaptive UCB: beta decreases from 2.0 to 0.5 to transition from exploration to exploitation
            beta = 2.0 * (1.0 - (i / self.budget)) + 0.5
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return -(mu - beta * np.maximum(sigma, 1e-9))

            # Optimize acquisition with multi-start
            best_acq_val = np.inf
            next_x = None
            
            # Combine random sampling with local optimization
            starts = np.vstack([np.random.rand(8, self.dim), best_x_norm.reshape(1, -1)])
            for x0 in starts:
                res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq_val:
                    best_acq_val, next_x = res.fun, res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb