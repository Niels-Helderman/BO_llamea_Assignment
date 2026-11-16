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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space filling
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        # Adaptive Beta schedule for UCB: explore more initially, exploit later
        beta_init = 2.0
        beta_final = 0.2
        
        for i in range(init_size, self.budget):
            try:
                # Update hyperparams only when beneficial for performance
                if i % 15 == 0:
                    self.gp.n_restarts_optimizer = 1
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                beta = beta_init - (beta_init - beta_final) * (i / self.budget)
                
                def get_ucb(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    # Minimize UCB: mu - beta * sigma
                    return mu - beta * sigma

                best_val = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start gradient-based optimization of acquisition
                for _ in range(8):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ucb, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
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