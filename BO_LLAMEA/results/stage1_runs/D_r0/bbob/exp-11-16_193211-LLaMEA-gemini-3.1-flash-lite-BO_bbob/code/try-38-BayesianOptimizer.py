import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD (Automatic Relevance Determination) by setting length_scale as a vector
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=0.0, 
            normalize_y=True, 
            n_restarts_optimizer=2,
            optimizer='fmin_l_bfgs_b'
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design: Sobol sequence for better coverage than random
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            # Refit GP
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                pass # Use previous hyperparameters if optimization fails
            
            # Acquisition: Expected Improvement
            def ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Candidates: best observed + random samples
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            for x0 in candidates:
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            y_new = func(next_x)
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x