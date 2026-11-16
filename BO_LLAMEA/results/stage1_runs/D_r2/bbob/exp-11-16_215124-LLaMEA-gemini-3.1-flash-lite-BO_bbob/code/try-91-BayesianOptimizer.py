import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD kernel to learn independent length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=False, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim + 2, 20), self.budget // 5)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        # Log-transform target for positive/heavy-tailed landscapes (standard practice)
        y_min = np.min(y)
        y_shift = abs(y_min) + 1.0
        y_trans = np.log(y - y_min + 1.0)
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            try:
                # Update transformation and refit GP
                y_min = np.min(y)
                y_trans = np.log(y - y_min + 1e-6)
                
                if i % 10 == 0:
                    self.gp.n_restarts_optimizer = 3
                else:
                    self.gp.n_restarts_optimizer = 0
                
                self.gp.fit(X, y_trans)
                
                def get_ei(x_norm):
                    x_norm = np.clip(x_norm, 0, 1)
                    mu_trans, std_trans = self.gp.predict((lb + x_norm * range_val).reshape(1, -1), return_std=True)
                    # Mapping EI back from log-space (empirical approx)
                    best_y_trans = np.min(y_trans)
                    Z = (best_y_trans - mu_trans) / (std_trans + 1e-10)
                    return -((best_y_trans - mu_trans) * norm.cdf(Z) + std_trans * norm.pdf(Z))

                # Multi-start acquisition optimization
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(get_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
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