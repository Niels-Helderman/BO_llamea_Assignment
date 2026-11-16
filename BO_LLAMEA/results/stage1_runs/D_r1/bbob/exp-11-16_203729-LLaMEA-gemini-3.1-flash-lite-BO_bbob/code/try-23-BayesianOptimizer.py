import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-7)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim + 2, 10)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        succ, fail = 0, 0

        def get_log_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-15, None)
            diff = y_min - mu
            z = diff / sigma
            # Log-EI is more numerically stable
            return diff * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            try:
                # Periodic GP re-fitting
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            best_acq = -np.inf
            next_x = np.random.uniform(lb_tr, ub_tr)
            
            # Multi-start acquisition optimization
            for _ in range(12):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: -get_log_ei(x, self.gp, y_min), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-4)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                succ, fail = succ + 1, 0
                if succ >= 2:
                    self.length = min(1.0, self.length * 1.5)
                    succ = 0
            else:
                fail += 1
                if fail >= 4:
                    self.length = max(0.05, self.length * 0.5)
                    fail = 0
            
        return best_y, best_x_norm * (ub - lb) + lb