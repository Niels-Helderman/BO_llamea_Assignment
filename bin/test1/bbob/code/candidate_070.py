import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is generally more robust for BBOB functions than 2.5
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling than pure LHS
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        # Log-EI is more numerically stable for minimization
        def get_log_ei(x_norm, best_y, gp):
            x = (lb + x_norm * range_val).reshape(1, -1)
            mu, std = gp.predict(x, return_std=True)
            std = np.maximum(std, 1e-12)
            diff = best_y - mu
            z = diff / std
            # Log of EI to prevent underflow
            return -(np.log(np.maximum(diff, 1e-20)) + np.log(norm.cdf(z)) + np.log(std) + norm.logpdf(z))

        for i in range(init_size, self.budget):
            try:
                # Periodic refitting
                if i % 15 == 0:
                    self.gp.n_restarts_optimizer = 2
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                best_ei = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start acquisition optimization: global vs local
                starts = [np.random.rand(self.dim)]
                if i > init_size:
                    starts.append((best_x - lb) / range_val + np.random.normal(0, 0.05, self.dim))
                
                for x0 in starts:
                    x0 = np.clip(x0, 0, 1)
                    res = minimize(get_log_ei, x0, args=(best_y, self.gp), 
                                   bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_ei:
                        best_ei = res.fun
                        next_x_norm = res.x
                
                next_x = lb + np.clip(next_x_norm, 0, 1) * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x