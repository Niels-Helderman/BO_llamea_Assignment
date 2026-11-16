import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        # alpha provides numerical stability for deterministic functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        
        # Latin Hypercube/Uniform sampling as fallback, prioritize density
        n_init = min(self.budget // 4, 15)
        for _ in range(n_init):
            x = np.random.uniform(lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = np.min(y_obs)
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodic hyperparameter optimization
            if (i - n_init) % 8 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            
            self.gp.fit(X, y)
            
            y_min = np.min(y)
            
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]
            
            # Use differential evolution for global acquisition optimization
            try:
                res = differential_evolution(acquisition, bounds=list(zip(lb, ub)), popsize=10, tol=1e-3, mutation=(0.5, 1))
                next_x = res.x
            except:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x