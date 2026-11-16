import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        # Using a slightly higher jitter (alpha) for numerical stability with deterministic functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling logic (simplified)
        n_init = min(self.budget, 2 * self.dim + 4)
        X_obs, y_obs = [], []
        
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) + i) / (n_init + 1)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Refit kernel hyperparameters every few steps
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Use LCB (Lower Confidence Bound) for balance, which is often more robust than EI
            kappa = 2.0
            def lcb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - kappa * sigma
            
            # Multi-start optimization of the acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            # Candidate points from best observed and random samples
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start in starts:
                res = minimize(lcb, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x