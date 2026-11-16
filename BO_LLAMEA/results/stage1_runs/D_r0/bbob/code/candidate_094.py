import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        # Using a fixed jitter to ensure numerical stability for deterministic functions
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-9,
            normalize_y=True,
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Larger initial design to capture objective landscape features early
        n_init = max(10, 5 * self.dim)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        idx = np.argmin(y_obs)
        best_x = np.array(X_obs[idx])
        best_y = float(y_obs[idx])
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to balance cost and accuracy
            self.gp.n_restarts_optimizer = 3 if i % 25 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI is more numerically stable than EI
                log_ei_val = np.log(diff + 1e-16) + norm.logcdf(z) + np.log1p(sigma/diff * norm.pdf(z)/norm.cdf(z))
                return -log_ei_val if np.isfinite(log_ei_val) else 0.0

            # Multi-start acquisition optimization to avoid local optima
            best_acq = np.inf
            next_x = None
            for _ in range(3):
                x0 = np.random.uniform(lb, ub)
                res = minimize(log_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x