import numpy as np
from scipy.optimize import minimize
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
        n_init = 2 * self.dim + 5
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs = []
        y_obs = []
        best_x = None
        best_y = np.inf
        
        for x in init_samples:
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            if y < best_y:
                best_y, best_x = float(y), x
        
        tr_size = 0.5
        tr_min = 1e-3
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-fit hyperparameters
            if i % 10 == 0:
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            beta = 2.0 * np.sqrt(2.0 * np.log((i + 1) * np.pi**2 / 1.2)) # Standard UCB exploration
            
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            def ucb(x):
                x = np.atleast_2d(x)
                mu, sigma = self.gp.predict(x, return_std=True)
                return mu - beta * sigma

            next_x = None
            best_acq = np.inf
            # Multi-start acquisition optimization
            for _ in range(5):
                x0 = np.random.uniform(tr_lb, tr_ub)
                res = minimize(ucb, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
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
                tr_size = min(0.8, tr_size * 1.2)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                if tr_size <= tr_min:
                    tr_size = 0.5 
                
        return float(best_y), best_x