import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 kernel with Automatic Relevance Determination (ARD)
        # WhiteKernel added for numerical stability (jitter)
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling for improved initial space coverage
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_init)
        X = qmc.scale(sample, lb, ub)
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
            # Fit GP: wrap in try-except to handle potential singular matrix issues
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Acquisition optimization using multi-start L-BFGS-B
            # Using 8 random points + 1 best point as starting candidates
            starts = np.vstack([np.random.uniform(lb, ub, (8, self.dim)), best_x])
            
            best_acq = np.inf
            next_x = None
            
            for x0 in starts:
                res = minimize(neg_ei, x0, args=(self.gp, y_min), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for numerical failures
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            val = func(next_x)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x