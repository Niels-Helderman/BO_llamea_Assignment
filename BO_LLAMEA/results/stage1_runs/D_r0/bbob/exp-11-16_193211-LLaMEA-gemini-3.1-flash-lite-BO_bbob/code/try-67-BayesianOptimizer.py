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
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = 2 * self.dim + 10
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = float(y_obs[best_idx])
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data, y_data = np.array(X_obs), np.array(y_obs)
            self.gp.n_restarts_optimizer = 2 if (i % 25 == 0) else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log of Expected Improvement to prevent underflow/exploding gradients
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            next_x = None
            # Local optimization within dynamic trust region
            res = minimize(log_ei, best_x, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
            if res.success:
                next_x = res.x
            else:
                next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = np.clip(tr_size * 1.2, 0.05, 1.0)
            else:
                tr_size = np.clip(tr_size * 0.8, 0.05, 1.0)
                
        return best_y, best_x