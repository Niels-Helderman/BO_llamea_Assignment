import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-12)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Initial sampling: 12 + 2*dim is sufficient to warm-start the GP
        n_init = 12 + 2 * self.dim
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = list(qmc.scale(sampler.random(n_init), lb, ub))
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)
            
            # Efficient hyperparameter tuning schedule
            self.gp.n_restarts_optimizer = 3 if i < 150 else 0
            try:
                self.gp.fit(X_arr, y_arr)
            except:
                pass
            
            # Log-EI acquisition function for numerical robustness
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                with np.errstate(divide='ignore'):
                    z = (best_y - mu) / sigma
                    return -( (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

            # Adaptive Trust Region
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            next_x = None
            best_acq = np.inf
            
            # Multi-start acquisition optimization
            starts = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(6)]
            for x0 in starts:
                res = minimize(log_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None: next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Update best and adjust TR based on success
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(0.02, tr_size * 0.85)
                
        return best_y, best_x