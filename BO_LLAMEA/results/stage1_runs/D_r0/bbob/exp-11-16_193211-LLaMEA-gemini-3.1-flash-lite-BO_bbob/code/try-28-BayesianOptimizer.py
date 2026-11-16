import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use WhiteKernel to handle deterministic noise and improve conditioning
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 2
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodic hyperparameter optimization every 20 steps
            fit_hyper = (i % 20 == 0)
            self.gp.optimizer = 'fmin_l_bfgs_b' if fit_hyper else None
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Log-EI Acquisition function
            def log_ei(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                with np.errstate(divide='ignore'):
                    diff = best_y - mu
                    z = diff / sigma
                    log_ei_val = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -log_ei_val.flatten()

            # Global optimization for acquisition
            next_x = None
            best_acq = np.inf
            # Multi-start optimization
            for x0 in [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for numerical failure
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x