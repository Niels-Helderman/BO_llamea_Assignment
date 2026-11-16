import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel to learn dimension-wise importance
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=False, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X, y = [], []
        # Initial sampling using Sobol sequence for better space filling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n=min(2 * self.dim + 4, self.budget // 4))
        init_points = lb + sample * (ub - lb)
        
        for x in init_points:
            X.append(x)
            y.append(func(x))
            
        X = np.array(X)
        y = np.array(y)
        best_idx = np.argmin(y)
        best_y, best_x = y[best_idx], X[best_idx]

        for i in range(len(X), self.budget):
            try:
                # Log-transform target for better numerical stability/skewness management
                y_min = np.min(y)
                y_target = np.log(y - y_min + 1e-9)
                
                self.gp.fit(X, y_target)
                
                def ei(x_norm):
                    x = lb + x_norm * (ub - lb)
                    mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                    # Mapping GP prediction back from log space
                    # Using Delta Method approximation for improvement
                    best_log = np.log(best_y - y_min + 1e-9)
                    Z = (best_log - mu) / (std + 1e-9)
                    return -((best_log - mu) * norm.cdf(Z) + (std + 1e-9) * norm.pdf(Z))

                # Multi-start acquisition
                candidates = [np.random.rand(self.dim)]
                # Add local search around the current best
                candidates.append((best_x - lb) / (ub - lb))
                
                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                for x0 in candidates:
                    res = minimize(ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * (ub - lb)
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x