import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        self.fit_interval = 20

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 2
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx].copy()
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            if i % self.fit_interval == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                pass
            
            # Log-EI acquisition function
            def neg_log_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI calculation for numerical stability
                log_ei = np.log(sigma + 1e-12) + (z * norm.cdf(z) + norm.pdf(z))
                return -log_ei

            next_x = None
            best_acq = np.inf
            
            # Multi-start acquisition optimization
            starts = np.vstack([best_x, qmc.scale(sampler.random(8), lb, ub)])
            for x0 in starts:
                res = minimize(neg_log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
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