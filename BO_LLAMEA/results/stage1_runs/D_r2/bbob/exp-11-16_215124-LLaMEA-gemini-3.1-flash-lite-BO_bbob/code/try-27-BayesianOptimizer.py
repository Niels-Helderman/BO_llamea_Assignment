import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

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
        
        init_size = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        refit_interval = 20
        
        for i in range(init_size, self.budget):
            try:
                if (i - init_size) % refit_interval == 0:
                    self.gp.n_restarts_optimizer = 3
                else:
                    self.gp.n_restarts_optimizer = 0
                
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x_norm = np.clip(x_norm, 0, 1)
                    mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x_norm = None
                tr_center = (best_x - lb) / range_val
                
                for _ in range(12):
                    tr_lb = np.maximum(tr_center - tr_len, 0)
                    tr_ub = np.minimum(tr_center + tr_len, 1)
                    x0 = np.random.uniform(tr_lb, tr_ub)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-3)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.2)
            else:
                tr_len *= 0.8
                if tr_len < 0.01: tr_len = 0.5
                
        return float(best_y), best_x