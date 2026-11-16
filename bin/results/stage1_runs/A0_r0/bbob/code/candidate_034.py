import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e1))
        # Use a small alpha for deterministic functions, but increase jitter as points grow to avoid singular matrices
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube-style initial seeding
        n_init = min(self.budget, 2 * self.dim + 4)
        for i in range(n_init):
            x = lb + (ub - lb) * (i + 0.5) / n_init # Simple deterministic grid seeding
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically update hyperparams; increase noise tolerance slightly as data grows
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                self.gp.kernel.set_params(**self.gp.kernel.get_params())
                
            y_min = np.min(y)
            
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                # UCB-EI hybrid: balances exploitation with uncertainty-based exploration
                beta = 2.0
                return mu - beta * sigma
            
            # Acquisition optimization using multiple starts
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            seeds = [best_x] + [np.random.uniform(lb, ub) for _ in range(5)]
            for start_point in seeds:
                res = minimize(acquisition, start_point, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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