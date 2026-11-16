import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a robust kernel with a fixed small jitter (noise) for stability in deterministic settings
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds='fixed') + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialize with LHS for better space coverage than pure Sobol
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        # Adaptive UCB beta: decaying schedule to shift from exploration to exploitation
        for i in range(n_init, self.budget):
            try:
                # Refit hyperparameters every few steps to reduce computational overhead
                if i % 5 == 0:
                    self.gp.fit(X, y)
                
                def ucb_acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    beta = 2.0 * np.sqrt(np.log(i + 1))
                    return mu - beta * sigma

                best_acq = np.inf
                next_x = np.random.rand(self.dim)
                
                # Multi-start acquisition optimization
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(ucb_acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            except Exception:
                # Fallback to random sampling if GP fitting or acquisition fails
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x * (ub - lb) + lb