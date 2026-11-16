import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is robust for COCO-BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        X_obs = []
        y_obs = []
        
        # Improved initial sampling using Sobol sequence for better space filling
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        init_points = lb + init_points * scale
        
        for x in init_points:
            if len(X_obs) < self.budget:
                y = func(x)
                X_obs.append(x)
                y_obs.append(y)
            
        for i in range(len(X_obs), self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically update hyperparams to save compute while maintaining accuracy
            try:
                self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 8 == 0) else None
                self.gp.fit(X, y)
            except:
                pass
                
            y_min = np.min(y)
            
            def ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                # Using standard EI formula
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Acquisition optimization with multi-start
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            
            # Candidate pool approach to avoid local optima
            pool = lb + np.random.uniform(0, 1, (20, self.dim)) * scale
            for start_x in pool:
                res = minimize(ei, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-6)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), X_obs[best_idx]