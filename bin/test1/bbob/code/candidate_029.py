import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Matern 5/2, constant variance, and WhiteKernel for numerical noise management
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space-filling initialization
        init_size = min(self.budget, max(2 * self.dim + 8, 20))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # GP refitting every step becomes expensive; only fit when data grows
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                best_ei = -np.inf
                next_x_norm = None
                
                # Multi-start optimization with increased density for better exploration
                tr_center = (best_x - lb) / range_val
                starts = [tr_center] + [np.random.uniform(0, 1, self.dim) for _ in range(11)]
                
                for x0 in starts:
                    tr_lb = np.maximum(0, tr_center - tr_len)
                    tr_ub = np.minimum(1, tr_center + tr_len)
                    res = minimize(get_ei, np.clip(x0, tr_lb, tr_ub), bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-5)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random search if GP/Optimization fails
                next_x = lb + np.random.uniform(0, 1, self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Trust region update logic
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len = max(0.01, tr_len * 0.85)
                
        return float(best_y), best_x