import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness properties, fixed white kernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Use Sobol sequence for better space-filling than pure random/LHS
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = max(2 * self.dim + 2, 16)
        X = lb + sampler.random(n=init_size) * scale
        y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(y)].copy()
        best_y = float(np.min(y))
        
        for i in range(init_size, self.budget):
            try:
                # Periodic hyperparameter tuning
                if i % 25 == 0:
                    self.gp.n_restarts_optimizer = 3
                else:
                    self.gp.n_restarts_optimizer = 0
                
                self.gp.fit(X, y)
                
                # Adaptive UCB: reduce beta over time to shift from exploration to exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                def ucb(x_norm):
                    x = lb + x_norm * scale
                    mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                    return mu - beta * std

                # Multi-start optimization of acquisition function
                best_val = np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * scale
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * scale
            
            val = func(next_x)
            if val < best_y:
                best_y, best_x = float(val), next_x
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
        return best_y, best_x