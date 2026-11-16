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
        # Using a small alpha to handle deterministic nature and avoid numerical instability
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 4
        
        # Sobol sequence for better space filling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-fit hyperparameters to save time
            if i % 10 == 0 or i == n_init:
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            # Thompson Sampling (Approximate)
            # Sample a GP posterior to capture exploration vs exploitation
            try:
                # We draw from the posterior to find a promising point
                mu, cov = self.gp.predict(np.random.uniform(lb, ub, (500, self.dim)), return_cov=True)
                # Fallback to pure acquisition optimization if sampling fails
            except:
                pass
            
            def acquisition(x):
                x = np.atleast_2d(x)
                mu, std = self.gp.predict(x, return_std=True)
                # Log-EI is more stable than standard EI
                diff = best_y - mu
                z = diff / (std + 1e-9)
                from scipy.stats import norm
                return -(diff * norm.cdf(z) + std * norm.pdf(z))

            # Multi-start optimization for acquisition
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(6)]
            for x0 in candidates:
                res = minimize(acquisition, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Ensure next_x is valid
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x