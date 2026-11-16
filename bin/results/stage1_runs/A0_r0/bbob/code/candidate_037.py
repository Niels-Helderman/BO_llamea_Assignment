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
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget // 5, 2 * self.dim + 5)
        X_obs, y_obs = [], []
        
        # Latin Hypercube Sampling variant for initial space filling
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) + i) / n_init
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X, y = np.array(X_obs), np.array(y_obs)
            
            # Periodic hyperparameter tuning
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 20 == 0 or i == n_init) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
                
            y_min = np.min(y)
            
            def log_ei_acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-10)
                diff = y_min - mu
                z = diff / sigma
                # Using log-EI for numerical stability in deep exploration
                log_ei = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z) + np.log(1 + (sigma/np.maximum(diff, 1e-20)) * np.exp(norm.logpdf(z) - norm.logcdf(z)))
                return -log_ei
            
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Multi-start optimization with greedy best and random candidates
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for s in candidates:
                res = minimize(log_ei_acq, s, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-5)
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