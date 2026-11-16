import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is robust for BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using scaled LHS for better coverage in 6D
        n_init = max(2 * self.dim + 4, 15)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def neg_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, None)
            diff = y_min - mu
            z = diff / sigma
            ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        for i in range(n_init, self.budget):
            try:
                # Fit every step, but using a faster non-restarting optimizer
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Acquisition optimization using multi-start L-BFGS-B
            best_acq = np.inf
            next_x = None
            
            # Propose from best known + random restarts to avoid local minima
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(8)]
            for x0 in starts:
                res = minimize(neg_ei, x0, args=(self.gp, y_min), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-6)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback/Exploration check: occasionally inject random point if acquisition is flat
            if next_x is None or np.random.rand() < 0.05:
                next_x = np.random.uniform(lb, ub, self.dim)
            
            val = func(next_x)
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x