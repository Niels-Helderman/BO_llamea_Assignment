import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(2 * self.dim + 2, 10)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        tr_min = 0.05
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit hyperparameters every 20 steps to save compute
            if (i - n_init) % 20 == 0:
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            beta = 1.96 * (1.0 - 0.5 * (i - n_init) / (self.budget - n_init))
            
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            def objective_acq(x):
                mu, sigma = self.gp.predict(np.atleast_2d(x), return_std=True)
                return (mu - beta * sigma).item()

            # Use differential evolution to robustly maximize acquisition
            res = differential_evolution(objective_acq, list(zip(tr_lb, tr_ub)), 
                                         popsize=10, mutation=(0.5, 1), recombination=0.7, tol=0.01)
            
            next_x = res.x if res.success or res.fun < 0 else np.random.uniform(tr_lb, tr_ub)
            y_new = func(next_x)
            
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                
        return best_y, best_x