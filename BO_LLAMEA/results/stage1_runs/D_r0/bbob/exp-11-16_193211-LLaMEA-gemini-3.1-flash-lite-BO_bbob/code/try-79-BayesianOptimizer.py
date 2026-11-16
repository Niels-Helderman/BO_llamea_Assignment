import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data, y_data = np.array(X_obs), np.array(y_obs)
            
            # Refit hyperparameters every 10 steps to balance cost and accuracy
            if i == n_init or i % 10 == 0:
                self.gp.n_restarts_optimizer = 2
                try: self.gp.fit(X_data, y_data)
                except: pass
            
            # EI Acquisition function
            mu_best = np.min(y_data)
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                z = (mu_best - mu) / sigma
                from scipy.stats import norm
                return -( (mu_best - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

            # Optimize acquisition in Trust Region
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            next_x = None
            best_acq = np.inf
            # Try multiple start points including center and boundaries
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(12)]
            for x0 in candidates:
                res = minimize(ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None: next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Adaptive trust region logic
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(0.05, tr_size * 0.85)
                
        return best_y, best_x