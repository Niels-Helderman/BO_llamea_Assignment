import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is robust for BBOB; WhiteKernel handles numeric jitter.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Increased initial samples to ensure better coverage for the first 30 steps.
        n_init = 2 * self.dim + 10
        X_obs, y_obs = [], []
        
        # Latin Hypercube Sampling-lite initialization
        for i in range(n_init):
            x = lb + (ub - lb) * (np.random.rand(self.dim) + i) / (n_init)
            x = np.clip(x, lb, ub)
            X_obs.append(x)
            y_obs.append(func(x))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        # Optimization loop
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            y_min = np.min(y_data)
            
            # Multi-start acquisition optimization
            best_acq = -1.0
            next_x = None
            
            # Propose points via random search + L-BFGS-B refinement
            for _ in range(5):
                x0 = lb + (ub - lb) * np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Fallback for numerical stability
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
                
        return best_y, best_x