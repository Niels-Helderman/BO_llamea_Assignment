import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            # Refit kernel periodically
            self.gp.n_restarts_optimizer = 1 if (i % 25 == 0) else 0
            try:
                self.gp.fit(X, y)
                
                # Adaptive Beta schedule for UCB (Exploration -> Exploitation)
                # Lowering beta allows focusing on local minima as budget depletes
                beta = 2.0 * np.sqrt(2.0 * np.log(i + 1)) * (1 - (i / self.budget)**0.5)
                
                def objective(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    # LCB: mu - beta * sigma
                    return mu - beta * sigma

                # Multi-start optimization of acquisition
                best_val = np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(6):
                    x0 = np.random.rand(self.dim)
                    res = minimize(objective, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x