import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel for BBOB: Matern 5/2 is standard for smooth-ish landscapes
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial space-filling design
        init_size = min(2 * self.dim + 2, self.budget // 4)
        X = np.random.uniform(0, 1, (init_size, self.dim))
        y = np.array([func(lb + x * range_val) for x in X])
        
        best_idx = np.argmin(y)
        best_x_norm, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        refit_freq = 5
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refitting to balance cost and accuracy
                if i % refit_freq == 0:
                    self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x_norm = None
                
                # Multi-start acquisition optimization within Trust Region
                for _ in range(8):
                    tr_lb = np.clip(best_x_norm - tr_len, 0, 1)
                    tr_ub = np.clip(best_x_norm + tr_len, 0, 1)
                    x0 = np.clip(np.random.normal(best_x_norm, tr_len/3), tr_lb, tr_ub)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                if next_x_norm is None: 
                    next_x_norm = np.random.uniform(0, 1, self.dim)
            except:
                next_x_norm = np.random.uniform(0, 1, self.dim)

            next_x = lb + next_x_norm * range_val
            val = func(next_x)
            X = np.vstack([X, next_x_norm])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = val, next_x_norm
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len *= 0.95
                if tr_len < 1e-3: 
                    tr_len = 0.5 # Reset TR if collapsed
                
        return float(best_y), (lb + best_x_norm * range_val)