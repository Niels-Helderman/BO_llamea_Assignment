import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling
        init_size = min(self.budget // 5, 30)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refitting to balance speed and accuracy
                self.gp.n_restarts_optimizer = 1 if i % 20 == 0 else 0
                self.gp.fit(X, y)
                
                # Adaptive UCB: Beta schedule
                beta = 2.0 * np.log(i + 1)
                
                # Thompson-style candidate sampling for acquisition optimization
                n_candidates = 500
                X_cand = lb + np.random.rand(n_candidates, self.dim) * range_val
                mu, sigma = self.gp.predict(X_cand, return_std=True)
                
                # UCB acquisition
                ucb = mu - beta * sigma
                next_x = X_cand[np.argmin(ucb)]
                
            except:
                # Fallback to random search if GP fitting fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            
            # Check for finite values to ensure stability
            if not np.isfinite(val):
                val = np.finfo(float).max
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x