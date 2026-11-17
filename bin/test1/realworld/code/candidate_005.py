import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        # Matern 5/2 is robust for most real-world black-box functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, n_restarts_optimizer=2, normalize_y=False)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Use Sobol sequence for space-filling initialization
        n_init = max(2 * self.dim, min(self.budget // 5, 20))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose_next(lb, ub)
            
            # Ensure x_next is finite and within bounds
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            
            self.X.append(x_next)
            self.y.append(y_val)
            
            # Refit periodically
            if len(self.y) >= n_init:
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                y_mean, y_std = np.mean(y_arr), np.std(y_arr) + 1e-9
                self.gp.fit(X_arr, (y_arr - y_mean) / y_std)

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub):
        y_min = np.min(self.y)
        y_mean, y_std = np.mean(self.y), np.std(self.y) + 1e-9
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            # Unnormalize mu for EI calculation
            mu = mu * y_std + y_mean
            std = std * y_std
            
            z = (y_min - mu) / (std + 1e-9)
            ei = (y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei[0]

        best_x = np.random.uniform(lb, ub, self.dim)
        best_ei = -np.inf
        
        # Random search to find good starting points for L-BFGS-B
        samples = np.random.uniform(lb, ub, (10, self.dim))
        for x0 in samples:
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        return best_x