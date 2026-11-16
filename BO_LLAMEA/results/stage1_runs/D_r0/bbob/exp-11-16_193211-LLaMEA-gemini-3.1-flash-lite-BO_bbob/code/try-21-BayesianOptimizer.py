import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel setup with a small noise floor
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design: Latin Hypercube Sampling is often more space-filling for high dimensions
        n_init = max(2 * self.dim, 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            # Periodically re-optimize hyperparameters only when data significantly grows
            if i % 20 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
                
            try:
                self.gp.fit(X_obs, y_obs)
            except:
                pass
            
            # Acquisition function: Expected Improvement
            # Using Log-EI to handle potentially large scales and prevent numerical underflow
            def ei_acq(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                with np.errstate(divide='ignore'):
                    z = diff / sigma
                    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # Minimize negative EI

            # Multi-start acquisition optimization
            best_acq_val = -np.inf
            next_x = None
            
            # Candidates: current best + random samples + points near current best
            candidates = [best_x]
            candidates.extend(np.random.uniform(lb, ub, (10, self.dim)))
            candidates.extend(np.clip(best_x + np.random.normal(0, 0.1 * (ub - lb), (5, self.dim)), lb, ub))
            
            for x0 in candidates:
                res = minimize(ei_acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_acq_val:
                    best_acq_val = -res.fun
                    next_x = res.x
            
            # Fallback if acquisition optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x