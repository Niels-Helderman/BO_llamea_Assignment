import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel to handle different scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 2
        
        # Use Sobol for better space-filling than LatinHypercube
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs = []
        y_obs = []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.4
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit kernel periodically with restarts for ARD optimization
            self.gp.n_restarts_optimizer = 2 if i % 20 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Expected Improvement acquisition
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            # Optimize acquisition within a Trust Region to favor local search
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            next_x = None
            best_acq = np.inf
            # Multi-start optimization
            for x0 in [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(5)]:
                res = minimize(ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.1)
            else:
                tr_size = max(0.05, tr_size * 0.9)
                
        return best_y, best_x