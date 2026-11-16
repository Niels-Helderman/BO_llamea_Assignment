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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube Sampling logic (simplified for reliability)
        n_init = min(self.budget, 2 * self.dim + 2)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.uniform(0, 1, self.dim) + i) / n_init
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X, y = np.array(X_obs), np.array(y_obs)
            
            # Refit kernels every 8 steps to balance cost vs accuracy
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                self.gp.kernel_ = self.kernel
                
            y_min = np.min(y)
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Global optimization of acquisition function via multi-start L-BFGS-B
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for start in candidates:
                res = minimize(ei, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x