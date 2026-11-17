import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-5)
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=False, n_restarts_optimizer=2)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X, y = [], []
        
        # Initial sampling: Latin Hypercube-like via Sobol or uniform if budget allows
        init_points = min(10, self.budget // 4)
        for _ in range(init_points):
            x = np.random.uniform(lb, ub)
            val = func(x)
            X.append(x)
            y.append(val)
            
        best_x = X[np.argmin(y)]
        best_y = np.min(y)
        
        while len(X) < self.budget:
            X_arr, y_arr = np.array(X), np.array(y)
            y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-8
            y_norm = (y_arr - y_mean) / y_std
            
            try:
                self.model.fit(X_arr, y_norm)
                
                def ei(x_norm):
                    x = lb + (ub - lb) * x_norm
                    mu, sigma = self.model.predict(x.reshape(1, -1), return_std=True)
                    mu = mu * y_std + y_mean
                    sigma = sigma * y_std
                    
                    target = np.min(y_norm) * y_std + y_mean
                    with np.errstate(divide='ignore'):
                        diff = target - mu
                        z = diff / (sigma + 1e-9)
                        from scipy.stats import norm
                        res = diff * norm.cdf(z) + sigma * norm.pdf(z)
                    return -res[0]

                res = minimize(ei, x0=np.random.rand(self.dim), bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                x_next = lb + (ub - lb) * res.x
            except:
                x_next = np.random.uniform(lb, ub)
            
            # Ensure no duplicates
            if np.any(np.all(np.isclose(X_arr, x_next), axis=1)):
                x_next = np.random.uniform(lb, ub)
                
            val = func(x_next)
            X.append(x_next)
            y.append(val)
            
            if val < best_y:
                best_y, best_x = val, x_next
                
        return float(best_y), best_x