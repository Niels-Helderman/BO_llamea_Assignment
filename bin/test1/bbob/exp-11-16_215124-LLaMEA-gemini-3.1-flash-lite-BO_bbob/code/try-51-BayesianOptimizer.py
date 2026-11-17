import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Constant scale + Matern 5/2 (smoothness) + Jitter for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling than pure random LHS
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = max(2 * self.dim + 2, 16)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(y)].copy()
        best_y = float(np.min(y))
        
        for i in range(init_size, self.budget):
            # Periodically re-optimize hyperparameters to balance cost/accuracy
            if i % 15 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X, y)
                # Adaptive Beta schedule: starts exploration, moves towards exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                def acq(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    return mu - beta * sigma
                
                # Multi-start optimization of the UCB acquisition function
                best_acq = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                for _ in range(10):
                    x0 = np.random.rand(self.dim)
                    res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val

            # Ensure we don't duplicate points which breaks matrix inversion
            if np.any(np.all(np.isclose(X, next_x), axis=1)):
                next_x = lb + np.random.rand(self.dim) * range_val
                
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x