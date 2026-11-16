import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=np.ones(dim), nu=2.5)
        # Using a fixed small alpha to handle deterministic nature effectively
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Sobol/LHC-like initialization: use 2*dim + 2 points for initial coverage
        n_init = min(self.budget, 2 * self.dim + 2)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Fit GP, refitting hyperparameters every few steps to reduce latency
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # EI acquisition function with numerical stability enhancements
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.clip(sigma, 1e-12, None)
                diff = y_min - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            # Multi-start acquisition optimization to avoid local optima
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start_point in candidates:
                res = minimize(acquisition, start_point, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
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