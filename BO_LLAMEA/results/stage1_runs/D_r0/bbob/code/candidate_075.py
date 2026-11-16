import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to learn dimension-specific length scales
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial space-filling using Sobol
        n_init = max(2 * self.dim, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = list(qmc.scale(sampler.random(n_init), lb, ub))
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Fit ARD GP with periodic hyperparameter updates
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Log-EI calculation for robust acquisition
            def neg_log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Using standard normal CDF/PDF for EI
                from scipy.stats import norm
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei if ei > 0 else 0

            # Multi-start optimization with a global component and a shrinking local one
            res_candidates = []
            for _ in range(5):
                # Mix random sampling with local exploitation around current best
                x0 = best_x + np.random.normal(0, 0.1 * (ub - lb) * (1 - i / self.budget), size=self.dim)
                x0 = np.clip(x0, lb, ub)
                res = minimize(neg_log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                res_candidates.append((res.fun, res.x))
            
            # Select the best acquisition point
            next_x = min(res_candidates, key=lambda x: x[0])[1]
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
                
        return best_y, best_x