import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e1), nu=2.5)
        # Using a slightly higher jitter to improve numerical stability in deterministic scenarios
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube-like initialization for better coverage
        n_init = min(self.budget, 2 * self.dim + 4)
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) if i > 0 else 0.5)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Efficient hyperparameter fitting schedule
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Lower Confidence Bound (LCB) for exploration/exploitation
            beta = 2.0 * np.sqrt(np.log(i + 1))
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            best_acq = np.inf
            next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            # Multi-start optimization of the acquisition function
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(5)]
            for start in candidates:
                res = minimize(acquisition, start, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
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