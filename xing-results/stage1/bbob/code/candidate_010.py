import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=0.0, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        n_init = min(self.budget, 2 * self.dim + 2)
        
        # Sobol sequence for better initial coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_sample = qmc.scale(sampler.random(n_init), lb, ub)
        
        best_y_curr = np.inf
        best_x_curr = None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_sample[i]
            else:
                if i % 3 == 0:  # Periodically update hyperparameters
                    self.gp.fit(np.array(X_obs), np.array(y_obs))
                
                best_y = np.min(y_obs)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                best_acq = np.inf
                x_next = np.random.uniform(lb, ub)
                # Multi-start optimization for acquisition function
                for _ in range(5):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
            
            y_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y_curr:
                best_y_curr = y_next
                best_x_curr = x_next
        
        return float(best_y_curr), np.array(best_x_curr)