import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0.0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(2 * self.dim, min(self.budget // 5, 10))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n_init)
        X_init = qmc.scale(sample, lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            y_obs.append(func(x))
            X_obs.append(x)
            
        best_idx = np.argmin(y_obs)
        best_x, best_y = X_obs[best_idx], y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 5 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            def ei(x):
                mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                std = np.maximum(std, 1e-12)
                z = (y_min - mu) / std
                return -( (y_min - mu) * norm.cdf(z) + std * norm.pdf(z) )

            # Multi-start acquisition optimization
            best_acq = 0
            next_x = np.random.uniform(lb, ub)
            for _ in range(8):
                start_x = np.random.uniform(lb, ub)
                res = minimize(ei, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                
        return float(best_y), best_x