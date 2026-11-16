import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        # Using a small alpha to handle deterministic nature with numerical robustness
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs, y_obs = [], []
        # Latin Hypercube-like initial design using Sobol-sequence or random uniform jitter
        n_init = min(self.budget // 4, 15 + self.dim)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x, best_y = X_obs[best_idx], y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X, y = np.array(X_obs), np.array(y_obs)
            # Re-optimize hyperparameters every few steps to balance cost and accuracy
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            def acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.clip(sigma, 1e-12, None)
                with np.errstate(divide='ignore'):
                    z = (y_min - mu) / sigma
                    ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]
            
            # Diverse multi-start strategy to avoid local optima in the acquisition surface
            best_acq = 0.0
            next_x = np.random.uniform(lb, ub)
            seeds = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for s in seeds:
                res = minimize(acq, s, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                
        return float(best_y), best_x