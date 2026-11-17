import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness and ConstantKernel to learn amplitude
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        # Using alpha for jitter to handle potential numerical instability in deterministic functions
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        # Initial design: use slightly more points for a 6D space to better inform the surrogate
        n_init = max(2 * self.dim + 2, 8)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_sample = qmc.scale(sampler.random(n_init), lb, ub)
        
        best_y_curr = np.inf
        best_x_curr = None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_sample[i]
            else:
                # Refit hyperparameters more frequently as data accumulates
                self.gp.fit(np.array(X_obs), np.array(y_obs))
                best_y = np.min(y_obs)
                
                # Acquisition function: Expected Improvement (EI)
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                best_acq = np.inf
                x_next = np.random.uniform(lb, ub)
                # Multi-start gradient-based acquisition optimization
                for _ in range(8):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), np.array(best_x_curr)