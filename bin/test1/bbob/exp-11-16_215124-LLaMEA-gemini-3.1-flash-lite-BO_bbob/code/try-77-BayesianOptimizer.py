import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use WhiteKernel as jitter for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial sampling
        init_size = max(2 * self.dim + 2, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            try:
                # Periodic hyperparameter update to balance overhead
                if i % 15 == 0:
                    self.gp.n_restarts_optimizer = 2
                else:
                    self.gp.n_restarts_optimizer = 0
                
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Hybrid acquisition optimization: 12 L-BFGS-B starts + 1000 random samples
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Random candidate pool for robustness
                cand_norm = np.random.rand(1000, self.dim)
                cands_ei = -np.array([get_ei(c) for c in cand_norm])
                best_ei = np.max(cands_ei)
                next_x_norm = cand_norm[np.argmax(cands_ei)]
                
                # Local refinement with L-BFGS-B
                for _ in range(4):
                    res = minimize(get_ei, next_x_norm, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            if val < best_y:
                best_y, best_x = float(val), next_x
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
                
        return best_y, best_x