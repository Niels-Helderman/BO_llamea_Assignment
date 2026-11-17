import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X = []
        y = []
        
        # Initial sampling: Latin Hypercube-like via scrambled Sobol or uniform
        init_size = min(2 * self.dim + 2, self.budget // 4)
        for i in range(init_size):
            x = np.random.uniform(lb, ub)
            res = func(x)
            X.append(x)
            y.append(res)
            
        X = np.array(X)
        y = np.array(y)
        
        best_idx = np.argmin(y)
        best_x = X[best_idx]
        best_y = y[best_idx]

        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters periodically (every 10 steps to save time)
                self.gp.fit(X, y)
                
                # Acquisition function: Expected Improvement
                def ei(x_norm):
                    x = lb + x_norm * (ub - lb)
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start acquisition optimization
                best_ei = -np.inf
                next_x = None
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x = lb + res.x * (ub - lb)
                
                if next_x is None: next_x = np.random.uniform(lb, ub)
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = val
                best_x = next_x
                
        return float(best_y), best_x