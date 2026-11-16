import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import qmc

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
        
        # Use Latin Hypercube Sampling for better initial coverage
        n_init = max(2 * self.dim + 2, self.budget // 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_samples = sampler.random(n=n_init)
        X_obs = qmc.scale(initial_samples, lb, ub)
        y_obs = [func(x) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize hyperparameters to handle growing data
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                self.gp.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
            
            # LCB acquisition: mu - kappa * sigma
            kappa = 2.0 * (1 - (i / self.budget))
            def acquisition(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - kappa * sigma
            
            # Multi-start optimization for the acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            seeds = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            for start in seeds:
                res = minimize(acquisition, start, bounds=list(zip(lb, ub)), method='L-BFGS-B')
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