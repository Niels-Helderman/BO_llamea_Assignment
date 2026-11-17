import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=True)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, y = [], []
        
        # Trust region parameters
        tr_center = None
        tr_size = 0.5
        best_y = np.inf
        best_x = None
        
        for i in range(self.budget):
            # Initialization
            if i < 2 * self.dim:
                x = np.random.uniform(lb, ub)
            else:
                # Acquisition optimization
                y_min = np.min(y)
                
                # Define trust region bounds
                if tr_center is not None:
                    low = np.maximum(0, (tr_center - tr_size) / (ub - lb))
                    high = np.minimum(1, (tr_center + tr_size) / (ub - lb))
                else:
                    low, high = 0.0, 1.0

                def acq(x_scaled):
                    x_raw = lb + x_scaled * (ub - lb)
                    mu, sigma = self.gp.predict(x_raw.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = y_min - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                best_acq = np.inf
                x_best_acq = np.random.rand(self.dim)
                for _ in range(4):
                    x0 = np.random.uniform(low, high)
                    res = minimize(acq, x0, bounds=list(zip(low, high)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_best_acq = res.fun, res.x
                x = lb + x_best_acq * (ub - lb)

            try:
                f = float(func(x))
            except:
                x = np.random.uniform(lb, ub)
                f = float(func(x))

            X.append(x)
            y.append(f)
            
            if f < best_y:
                best_y, best_x = f, x
                tr_center = x
                tr_size = min(1.0, tr_size * 1.1)
            else:
                tr_size *= 0.95
                if tr_size < 1e-3:
                    tr_center = None
                    tr_size = 0.5
            
            # Refit GP
            if len(X) >= 5 and (len(X) < 20 or len(X) % 10 == 0):
                try:
                    self.gp.fit(np.array(X), np.array(y))
                except:
                    pass
        
        return float(best_y), np.array(best_x)