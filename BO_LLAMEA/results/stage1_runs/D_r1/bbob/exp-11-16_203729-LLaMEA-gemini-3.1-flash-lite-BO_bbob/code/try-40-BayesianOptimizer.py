import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        # Using a small alpha to prevent inversion issues; normalize_y is key for BBOB functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.tr_length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Latin Hypercube-like spread (or simple uniform for simplicity in this constraints)
        n_init = max(self.dim * 3, 12)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        success_counter, failure_counter = 0, 0

        def get_neg_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e10)
            diff = y_min - mu
            z = diff / sigma
            ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            # Define Trust Region bounds
            lb_tr = np.clip(best_x_norm - self.tr_length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.tr_length / 2, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Optimization of acquisition: combine multi-start L-BFGS-B with TR constraints
            for _ in range(12):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(get_neg_ei, x0, args=(self.gp, y_min),
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-5)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback
            if next_x is None: 
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                success_counter += 1
                failure_counter = 0
                if success_counter >= 2:
                    self.tr_length = min(1.0, self.tr_length * 1.2)
                    success_counter = 0
            else:
                failure_counter += 1
                if failure_counter >= 3:
                    self.tr_length = max(0.02, self.tr_length * 0.5)
                    failure_counter = 0
            
            if self.tr_length < 0.05:
                self.tr_length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb