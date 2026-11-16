import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs, y_obs = [], []
        n_init = min(20, self.budget // 5)
        
        # Initial sampling
        for _ in range(n_init):
            x = lb + (ub - lb) * np.random.rand(self.dim)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = y_obs[best_idx]
        
        # Trust region parameters
        tr_scale = 0.5
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit periodically
            if (i - n_init) % 8 == 0:
                self.gp.fit(X_data, y_data)
            
            # Trust region bounds
            tr_lb = np.maximum(lb, best_x - tr_scale * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_scale * (ub - lb))
            
            y_min = np.min(y_data)
            next_x = None
            
            # Maximize EI within TR
            for _ in range(5):
                x0 = tr_lb + (tr_ub - tr_lb) * np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if next_x is None or -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            if next_x is None: next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                tr_scale = min(0.5, tr_scale * 1.2)
            else:
                tr_scale *= 0.9
                if tr_scale < 1e-3: tr_scale = 0.5
                
        return float(best_y), best_x