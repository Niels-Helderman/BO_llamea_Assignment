import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        # Initial sampling: Latin Hypercube-like spread using Sobol-sequence or random
        n_init = min(self.budget // 4, 2 * self.dim + 2)
        for i in range(n_init):
            x = np.random.uniform(lb, ub)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodic refitting
            try:
                if i % 10 == 0:
                    self.gp.optimizer = 'fmin_l_bfgs_b'
                else:
                    self.gp.optimizer = None
                self.gp.fit(X, y)
            except:
                pass
                
            # Acquisition function: Expected Improvement
            y_min = np.min(y)
            
            def ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                imp = y_min - mu
                z = imp / sigma
                return -(imp * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Global optimization of acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            for _ in range(5):
                start_x = np.random.uniform(lb, ub)
                res = minimize(ei, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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