import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-9, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, 2 * self.dim + 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n_init)
        X_init = qmc.scale(sample, lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically update hyperparameters, otherwise use current kernel
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Lower Confidence Bound: balance local search and exploration
            beta = 2.0
            def lcb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma

            best_acq = np.inf
            next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            # Multi-start optimization of acquisition function
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(4)]
            for start in candidates:
                res = minimize(lcb, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x