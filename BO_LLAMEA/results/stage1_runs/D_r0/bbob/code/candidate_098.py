import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit kernel hyperparams every 25 steps to save time
            self.gp.n_restarts_optimizer = 2 if i % 25 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Adaptive UCB: beta decreases to focus on exploitation
            beta = 2.0 * np.log((i + 1) * (self.dim + 1))
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Try global search from random candidates and local search from best_x
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(3)]
            for start in starts:
                res = minimize(ucb, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization failed
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x