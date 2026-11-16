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
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space filling
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = float(y_obs[best_idx])
        
        # Acquisition function: Expected Improvement
        def ei(x, model, y_min):
            mu, sigma = model.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)).flatten()

        for i in range(n_init, self.budget):
            try:
                # Refit kernel every 10 steps to save cost
                self.gp.n_restarts_optimizer = 2 if i % 10 == 0 else 0
                self.gp.fit(X_obs, y_obs)
            except:
                pass
            
            # Multi-start acquisition optimization
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(3)]
            best_acq = np.inf
            next_x = best_x
            
            for start_x in starts:
                res = minimize(
                    lambda x: ei(x, self.gp, best_y),
                    start_x,
                    bounds=list(zip(lb, ub)),
                    method='L-BFGS-B'
                )
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for numerical instability
            if np.any(np.isnan(next_x)):
                next_x = np.random.uniform(lb, ub)
                
            y_new = func(next_x)
            
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
                
        return best_y, best_x