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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_dim = ub - lb
        n_init = 2 * self.dim + 4
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Adaptive beta strategy
            beta = 2.0 * np.sqrt(2 * np.log(i + 1))
            
            # Trust region bounds
            tr_lb = np.maximum(lb, best_x - tr_size * range_dim)
            tr_ub = np.minimum(ub, best_x + tr_size * range_dim)
            
            def objective(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return (mu - beta * sigma).item()

            # Acquisition optimization with multistart
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(objective, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
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
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(0.01, tr_size * 0.95)
                # Periodically reset TR if stuck
                if i % 50 == 0:
                    tr_size = 0.5
                
        return best_y, best_x