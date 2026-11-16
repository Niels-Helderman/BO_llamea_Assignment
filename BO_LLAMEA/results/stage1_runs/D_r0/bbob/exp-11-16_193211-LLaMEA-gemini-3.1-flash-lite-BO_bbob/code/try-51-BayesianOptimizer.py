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
        n_init = 2 * self.dim + 4
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit hyperparameters for efficiency
            if (i - n_init) % 10 == 0:
                self.gp.kernel = self.kernel
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            # Decaying beta for UCB
            beta = 2.0 * np.sqrt(2.0 * np.log(i + 1))
            
            def ucb(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization using L-BFGS-B (robust local search)
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(6)]
            
            for x0 in candidates:
                res = minimize(ucb, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            else:
                # Add small jitter to avoid stagnation
                if np.random.rand() < 0.05:
                    next_x += np.random.normal(0, 0.01 * (ub - lb))
                    next_x = np.clip(next_x, lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x