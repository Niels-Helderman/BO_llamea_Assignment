import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD kernel: length_scale is an array of dim, allowing axis-specific length scales.
        # Log-transforming output makes the GP more robust to skewness in BBOB functions.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=False, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 5
        
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [float(func(x)) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Use log-transform to handle potentially non-Gaussian objective distributions
            y_log = np.log(y_data - np.min(y_data) + 1e-9)
            
            try:
                self.gp.fit(X_data, y_log)
            except:
                self.gp.kernel.set_params(k1=ConstantKernel(1.0))
            
            # Acquisition function: Expected Improvement
            y_min = np.min(y_log)
            def ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                from scipy.stats import norm
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            # Optimize EI with multi-start
            next_x = None
            best_acq = np.inf
            for _ in range(5):
                x0 = np.random.uniform(lb, ub)
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = float(func(next_x))
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                
        return best_y, best_x