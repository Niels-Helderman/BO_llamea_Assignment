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

    def _log_ei(self, x, gp, y_min):
        x = np.atleast_2d(x)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-12, None)
        with np.errstate(divide='ignore'):
            z = (y_min - mu) / sigma
            log_ei = np.log(sigma) + norm.logcdf(z) + np.log(norm.pdf(z) + (z * norm.cdf(z)))
        return -log_ei.flatten()

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, 10)
        
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically update hyperparameters: every 10 steps or if early
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 10 == 0 or i < n_init + 5) else None
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            next_x = None
            y_min = np.min(y_data)
            
            # Multi-start acquisition optimization
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(10)]
            best_acq = np.inf
            for x0 in candidates:
                res = minimize(lambda x: self._log_ei(x, self.gp, y_min), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x