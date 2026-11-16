import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is well-suited for BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling than random or plain LHS
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim + 2, 20), self.budget // 4)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refitting with jitter for numerical stability
                self.gp.n_restarts_optimizer = 1 if (i % 30 == 0) else 0
                self.gp.fit(X, y)
                
                # UCB acquisition: beta decays to transition from exploration to exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                def objective(x_norm):
                    mu, sigma = self.gp.predict((lb + x_norm * range_val).reshape(1, -1), return_std=True)
                    return (mu - beta * sigma).item()

                best_val = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start local optimization to find acquisition minimum
                for _ in range(8):
                    x0 = np.random.rand(self.dim)
                    res = minimize(objective, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random point if GP optimization fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x