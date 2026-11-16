import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smooth but realistic target functions; ConstantKernel handles scale.
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=2,
            optimizer='fmin_l_bfgs_b'
        )

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, y = [], []
        
        # Initial points: Space-filling strategy
        n_init = max(2 * self.dim, min(self.budget // 4, 12))
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * np.random.rand(self.dim)
            else:
                x = self._propose_next(np.array(X), np.array(y), lb, ub)
            
            val = func(x)
            X.append(x)
            y.append(val)
            
            # Periodically update the GP
            if len(X) >= 5:
                self.gp.fit(np.array(X), np.array(y))
        
        best_idx = np.argmin(y)
        return float(y[best_idx]), np.array(X[best_idx])

    def _propose_next(self, X, y, lb, ub):
        y_min = np.min(y)
        def acquisition(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            std = np.maximum(std, 1e-12)
            z = (y_min - mu) / std
            ei = (y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei[0]

        best_x = np.random.uniform(lb, ub)
        best_acq = np.inf
        
        # Multi-start acquisition optimization
        for _ in range(10):
            x0 = lb + (ub - lb) * np.random.rand(self.dim)
            res = minimize(acquisition, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        return best_x