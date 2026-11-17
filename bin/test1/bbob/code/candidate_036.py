import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for robust space filling
        init_size = min(self.budget // 10, 30)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refit to balance cost vs accuracy
                self.gp.fit(X, y)
                
                def get_neg_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start optimization for Acquisition Function
                best_ei = -np.inf
                next_x_norm = None
                
                # Trust region strategy
                tr_center = (best_x - lb) / range_val
                starts = [tr_center] + [np.clip(tr_center + np.random.normal(0, tr_len, self.dim), 0, 1) for _ in range(4)]
                
                for x0 in starts:
                    res = minimize(get_neg_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                    if -res.fun > best_ei:
                        best_ei, next_x_norm = -res.fun, res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Adaptive logic
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.01, tr_len * 0.9)
                
        return float(best_y), best_x