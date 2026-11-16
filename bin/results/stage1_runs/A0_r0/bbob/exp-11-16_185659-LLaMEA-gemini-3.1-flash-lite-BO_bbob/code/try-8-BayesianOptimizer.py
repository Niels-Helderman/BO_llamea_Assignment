import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness and ConstantKernel for amplitude scaling
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Increased initial sampling for better GP coverage in 6D
        n_init = max(2 * self.dim + 1, min(self.budget // 5, 12))
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodic refitting to save computation while maintaining model accuracy
            if (i - n_init) % 5 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            
            try:
                self.gp.fit(X, y)
            except:
                pass
                
            y_min = np.min(y)
            
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                with np.errstate(divide='ignore'):
                    z = (y_min - mu) / sigma
                    ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            for start_point in [best_x] + [np.random.uniform(lb, ub) for _ in range(3)]:
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