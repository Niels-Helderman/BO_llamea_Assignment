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
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodic hyperparameter tuning to reduce overhead
            self.gp.n_restarts_optimizer = 3 if i % 50 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Adaptive beta: decay from 2.0 to 0.5 to transition from exploration to exploitation
            beta = 2.0 - 1.5 * (i / self.budget)
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return float(mu - beta * sigma)

            # Multistart L-BFGS to find acquisition maximum
            best_acq = np.inf
            next_x = X_obs[np.argmin(y_obs)] # Fallback
            
            for _ in range(5):
                start_x = np.random.uniform(lb, ub)
                res = minimize(ucb, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Ensure next_x is strictly within bounds
            next_x = np.clip(next_x, lb, ub)
            
            y_new = func(next_x)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x