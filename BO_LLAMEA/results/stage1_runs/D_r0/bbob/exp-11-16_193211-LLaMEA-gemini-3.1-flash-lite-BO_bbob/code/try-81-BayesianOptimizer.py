import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space filling
        n_init = max(2 * self.dim + 1, 15)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodical hyperparameter tuning
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def log_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI: log(diff * Phi(z) + sigma * phi(z))
                # Using logsumexp trick for stability
                return -(np.log(np.maximum(sigma, 1e-20)) + np.log(norm.cdf(z) * z + norm.pdf(z)))

            # Acquisition optimization with multi-start local search
            next_x = None
            best_acq = np.inf
            
            # Start from best point and random points
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x