import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 4 * self.dim
        
        # LHS initialization for better space filling
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(np.array(x)) for x in X_obs]
        
        best_x = np.array(X_obs[np.argmin(y_obs)])
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit kernel hyperparameters every 10 steps to save time
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 10 == 0) else None
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            
            def log_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                # Log-EI for numerical stability
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Multi-start acquisition optimization
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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
                
        return best_y, best_x