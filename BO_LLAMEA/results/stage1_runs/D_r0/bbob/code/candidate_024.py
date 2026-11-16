import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel configuration with small jitter for deterministic surface smoothness
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=2
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 4
        
        # Sobol sequence for better space-filling initial coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        tr_min = 1e-4
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit GP hyperparameters; limited optimization
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Decaying UCB exploration parameter
            beta = 2.5 * (1.0 - (i / self.budget))
            
            # Adaptive Trust Region (TR) constraints
            tr_lb = np.clip(best_x - tr_size * (ub - lb), lb, ub)
            tr_ub = np.clip(best_x + tr_size * (ub - lb), lb, ub)
            
            def acq(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization
            next_x = None
            best_val = np.inf
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(10)]
            for x0 in candidates:
                res = minimize(acq, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_val:
                    best_val, next_x = res.fun, res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Update incumbent and adjust trust region
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                if tr_size <= tr_min:
                    tr_size = 0.5
                
        return best_y, best_x