import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is generally robust for BBOB, WhiteKernel adds numerical robustness
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Larger initial design for better surrogate conditioning
        n_init = 4 * self.dim
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodic hyperparameter optimization
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 3
                self.gp.fit(X_data, y_data)
            else:
                self.gp.fit(X_data, y_data)
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log of EI is numerically more stable for optimization
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            # Adaptive search bounds
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            for _ in range(5):
                x0 = np.random.uniform(tr_lb, tr_ub)
                res = minimize(log_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = np.clip(tr_size * 1.2, 0.05, 1.0)
            else:
                tr_size = np.clip(tr_size * 0.8, 0.05, 1.0)
                
        return best_y, best_x