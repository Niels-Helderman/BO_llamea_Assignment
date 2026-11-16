import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space filling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = 2 * self.dim + 2
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_radius = 0.5
        
        for i in range(n_init, self.budget):
            X_data, y_data = np.array(X_obs), np.array(y_obs)
            
            # Periodically re-optimize hyperparameters
            if i % 20 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # UCB exploration parameter
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            # Trust region boundaries
            tr_lb = np.maximum(lb, best_x - tr_radius * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_radius * (ub - lb))
            
            # Local acquisition optimization (CMA-ES style simplified: sampling & selection)
            pop_size = 20
            candidates = np.random.normal(best_x, tr_radius * (ub - lb) * 0.2, (pop_size, self.dim))
            candidates = np.clip(candidates, tr_lb, tr_ub)
            
            mu, sigma = self.gp.predict(candidates, return_std=True)
            acq = mu - beta * sigma
            next_x = candidates[np.argmin(acq)]
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Update best and adaptive trust region
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_radius = min(1.0, tr_radius * 1.1)
            else:
                tr_radius = max(0.01, tr_radius * 0.95)
                
        return best_y, best_x