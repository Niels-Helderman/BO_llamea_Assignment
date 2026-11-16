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
        range_val = ub - lb
        
        # Larger robust initial design using Sobol-LHS hybrid
        n_init = 2 * self.dim + 20
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        stagnation_counter = 0
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Efficient hyperparameter fitting frequency
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 1
                self.gp.fit(X_data, y_data)
            
            # Log-EI for stable acquisition
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                from scipy.stats import norm
                # Log(EI) = log(diff * Phi(z) + sigma * phi(z))
                # Using logsumexp trick: log(exp(a) + exp(b))
                log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * z**2
                log_Phi = norm.logcdf(z)
                return -np.logaddexp(np.log(diff) + log_Phi, np.log(sigma) + log_phi)

            # Adaptive Trust Region
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            # Multi-start optimization inside TR
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(5)]
            next_x, best_acq = None, np.inf
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None: next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
                stagnation_counter = 0
            else:
                tr_size = max(0.05, tr_size * 0.8)
                stagnation_counter += 1
                if stagnation_counter > 15: # Reset TR on stagnation
                    tr_size = 0.5
                    stagnation_counter = 0
                
        return best_y, best_x