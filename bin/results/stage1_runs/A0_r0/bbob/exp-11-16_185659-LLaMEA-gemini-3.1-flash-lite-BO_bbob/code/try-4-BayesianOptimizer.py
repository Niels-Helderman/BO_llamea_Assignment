import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        X_obs, y_obs = [], []
        n_init = max(2 * self.dim + 2, min(self.budget // 5, 12))
        
        # Sobol sequence for better space-filling initialization
        sampler = Sobol(d=self.dim, scramble=True)
        initial_samples = sampler.random(n=n_init)
        
        for i in range(n_init):
            x = lb + initial_samples[i] * range_val
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        best_x, best_y = X_obs[best_idx].copy(), y_obs[best_idx]
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize kernel hyperparameters
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
            self.gp.fit(X, y)
            
            # UCB acquisition: mu - kappa * sigma
            kappa = 2.0 * np.sqrt(np.log(i + 1))
            
            def ucb(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                return mu - kappa * np.maximum(sigma, 1e-12)
            
            # Multi-start local optimization for acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            
            # Candidates: mix of random sampling and local search
            candidates = np.random.uniform(lb, ub, (5, self.dim))
            for start_x in candidates:
                res = minimize(ucb, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                
        return float(best_y), best_x