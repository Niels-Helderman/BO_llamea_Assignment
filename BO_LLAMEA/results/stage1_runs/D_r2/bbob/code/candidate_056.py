import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoother functions, WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling than pure LHS
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters every 20 iterations or if data is small
                if i % 20 == 0 or i < init_size + 5:
                    self.gp.n_restarts_optimizer = 1
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                # Acquisition: Use Log-EI to prevent underflow, add UCB component for exploration
                def acquisition(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    mu, sigma = self.gp.predict(x, return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    
                    # Log-EI calculation
                    with np.errstate(divide='ignore'):
                        diff = best_y - mu
                        z = diff / sigma
                        log_ei = np.log(diff) + norm.logcdf(z) + np.log(1 + (sigma / diff) * (norm.pdf(z) / norm.cdf(z)))
                    return -np.exp(log_ei)

                # Multi-start acquisition optimization
                best_acq = np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Search using random candidates followed by L-BFGS-B refinement
                candidates = np.random.rand(20, self.dim)
                for x0 in candidates:
                    res = minimize(acquisition, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x