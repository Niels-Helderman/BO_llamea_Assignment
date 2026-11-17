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
        self.X = []
        self.y = []
        # Use Matern 5/2 as it provides a good balance between smoothness and flexibility
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(2 * self.dim + 2, self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose_next(lb, ub)
            
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            
            self.X.append(x_next)
            self.y.append(y_val)
            
            if len(self.y) >= n_init:
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                # Standardize y based on observations to maintain numerical stability
                self.y_mean, self.y_std = np.mean(y_arr), np.std(y_arr) + 1e-9
                y_norm = (y_arr - self.y_mean) / self.y_std
                # Periodically re-fit hyperparameters
                if i % 5 == 0 or i == n_init:
                    self.gp.fit(X_arr, y_norm)
                    self.gp.kernel_ = self.gp.kernel_

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub):
        y_min = np.min(self.y)
        
        def acq(x):
            mu_norm, std_norm = self.gp.predict(x.reshape(1, -1), return_std=True)
            mu = mu_norm * self.y_std + self.y_mean
            std = std_norm * self.y_std
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (std + 1e-12)
                ei = (y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei[0]

        # Use multiple random restarts for acquisition optimization
        best_x = np.random.uniform(lb, ub, self.dim)
        best_ei = -np.inf
        
        candidates = [np.random.uniform(lb, ub, self.dim) for _ in range(8)]
        # Add current best as a starting point
        candidates.append(self.X[np.argmin(self.y)])
        
        for x0 in candidates:
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        return best_x