import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0.0, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )
        self.fit_interval = 20

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial design
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = 2 * self.dim + 2
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters to save compute
            if (i - n_init) % self.fit_interval == 0:
                self.gp.n_restarts_optimizer = 5
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                pass
            
            # Log-Expected Improvement Acquisition
            def log_ei(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Using log-space to improve numerical stability for EI
                log_ei_val = np.log(sigma) + norm.logcdf(z) + np.log(1 + np.exp(np.log(diff) - np.log(sigma) - np.log(norm.cdf(z) + 1e-20)))
                return -log_ei_val # Negative for minimization

            # Multi-start acquisition optimization
            next_x = None
            best_acq = np.inf
            seeds = np.vstack([best_x, qmc.scale(sampler.random(10), lb, ub)])
            for x0 in seeds:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x