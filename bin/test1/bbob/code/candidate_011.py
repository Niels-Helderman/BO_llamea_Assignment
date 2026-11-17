import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Latin Hypercube Sampling (LHS) for initial design
        init_size = max(2 * self.dim + 4, 15)
        X = []
        for i in range(init_size):
            x = lb + (np.random.rand(self.dim) + np.random.permutation(init_size)[i] / init_size) * range_val / 2
            X.append(np.clip(x, lb, ub))
        
        y = np.array([func(x) for x in X])
        X = np.array(X)
        best_x = X[np.argmin(y)].copy()
        best_y = np.min(y)
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refitting
                if i % 8 == 0:
                    self.gp.fit(X, y)
                
                # EI Acquisition
                def get_ei(x_norm):
                    x = lb + np.clip(x_norm, 0, 1) * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x = None
                
                # Multi-start acquisition optimization with Trust Region
                tr_center = (best_x - lb) / range_val
                for _ in range(8):
                    tr_lb = np.clip(tr_center - tr_len, 0, 1)
                    tr_ub = np.clip(tr_center + tr_len, 0, 1)
                    x0 = np.clip(np.random.normal(tr_center, tr_len/3), tr_lb, tr_ub)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x = lb + res.x * range_val
                
                if next_x is None: next_x = np.random.uniform(lb, ub)
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len *= 0.95
                if tr_len < 0.05: tr_len = 0.5
        
        return float(best_y), best_x