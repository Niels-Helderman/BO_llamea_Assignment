import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel to learn dimension-specific scales
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial space-filling using Sobol
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = list(qmc.scale(sampler.random(n_init), lb, ub))
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to balance cost/accuracy
            if i < 20 or i % 30 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            
            # Acquisition function: Log-Expected Improvement
            def log_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                # Log EI: log(diff * Phi(z) + sigma * phi(z))
                # Using stability-improved calculation
                log_ei_val = np.log(np.maximum(sigma * (z * norm.cdf(z) + norm.pdf(z)), 1e-20))
                return -log_ei_val # Negative for minimization

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Candidates: mix of best incumbent and uniform random samples
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x