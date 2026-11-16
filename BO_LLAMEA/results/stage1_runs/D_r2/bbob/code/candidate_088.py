import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel with fixed jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space filling than random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim, 16), self.budget // 4)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        # Adaptive UCB schedule: starts high for exploration, cools for exploitation
        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters every 32 steps to save costs
                if i % 32 == 0:
                    self.gp.n_restarts_optimizer = 1
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                beta = 2.0 * np.sqrt(np.log(i + 1)) # Standard UCB exploration factor
                
                def get_ucb(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    # UCB is mu - beta * sigma for minimization
                    return (mu - beta * sigma).item()

                # Multi-start acquisition optimization
                best_acq = np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(6):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback: Sobol sample if GP fails
                next_x = lb + sampler.random(1).flatten() * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x