import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a stable kernel setup with constant scale and Matern 5/2
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial coverage
        n_init = max(2, min(self.budget // 5, 2 * self.dim + 2))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n_init)
        X_init = qmc.scale(sample, lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            y_obs.append(func(x))
            X_obs.append(x)
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to balance cost/accuracy
            optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            self.gp.optimizer = optimizer
            self.gp.fit(X, y)
            
            y_min = np.min(y)
            
            # Acquisition: Expected Improvement
            def acq(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # Minimize negative EI
            
            # Multi-start local optimization to find acquisition global maximum
            best_acq = 1.0
            next_x = np.random.uniform(lb, ub)
            # Candidate pool to avoid getting stuck in local optima
            candidates = np.random.uniform(lb, ub, (10, self.dim))
            for start_x in candidates:
                res = minimize(acq, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-6)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), np.array(best_x)