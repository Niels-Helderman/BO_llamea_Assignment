import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is robust for BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initial sampling density using Latin Hypercube
        n_init = min(2 * self.dim + 2, self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_init)
        X_init = qmc.scale(sample, lb, ub)
        
        X_obs = []
        y_obs = []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_idx = np.argmin(y_obs)
        best_x = X_obs[best_idx]
        best_y = y_obs[best_idx]

        def get_ei(x, gp, y_min):
            with np.errstate(divide='ignore'):
                mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_min - mu) / sigma
                return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        # Main optimization loop
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit hyperparameters every 8 steps to balance cost and model accuracy
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            best_acq = -np.inf
            next_x = None
            
            # Multi-start optimization of EI
            for _ in range(8):
                x0 = lb + (ub - lb) * np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Fallback if acquisition optimization fails
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            try:
                y_new = func(next_x)
            except:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
                y_new = func(next_x)

            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = next_x
                
        return float(best_y), best_x