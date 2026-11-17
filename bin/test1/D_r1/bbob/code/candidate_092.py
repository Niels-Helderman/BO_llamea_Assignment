import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small jitter to handle deterministic objective noise/flat regions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=0.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial Design
        n_init = min(self.budget // 5, 20)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Adaptive UCB beta schedule: start high for exploration, decay for exploitation
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                beta = 2.0 * np.sqrt(np.log(i + 1))
            except:
                # If GP fitting fails, use random sampling
                next_x = np.random.rand(self.dim)
            else:
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    return mu - beta * sigma

                # Multi-start acquisition optimization
                candidates = np.random.rand(10, self.dim)
                best_acq_val = np.inf
                next_x = candidates[0]
                
                for x0 in candidates:
                    res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        next_x = res.x
            
            # Ensure boundaries
            next_x = np.clip(next_x, 0, 1)
            
            # Execute objective
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                
        return best_y, best_x_norm * (ub - lb) + lb