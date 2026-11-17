import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel with Automatic Relevance Determination
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2)
        self.y_log = False

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        X_obs = []
        y_obs = []
        
        n_init = max(2, min(self.budget // 4, 2 * self.dim))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._acquire(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y, best_x = y, x
                
        return best_y, best_x

    def _acquire(self, X, y, lb, ub):
        # Apply log transform if values are positive to handle heavy-tailed BBOB landscapes
        y_trans = np.log(y - np.min(y) + 1e-6)
        X_norm = (X - lb) / (ub - lb)
        
        try:
            self.gp.fit(X_norm, y_trans)
        except:
            return np.random.uniform(lb, ub)
        
        # Thompson Sampling / Expected Improvement approximation via Monte Carlo
        n_candidates = 500
        candidates = np.random.rand(n_candidates, self.dim)
        
        # Generate a sample from the GP posterior at candidate points
        mu, cov = self.gp.predict(candidates, return_cov=True)
        # Add jitter for numerical stability
        eps = np.random.normal(0, 1, n_candidates)
        # Cholesky decomposition of posterior covariance
        try:
            L = np.linalg.cholesky(cov + np.eye(n_candidates) * 1e-6)
            f_sample = mu + L @ eps
            idx = np.argmin(f_sample)
        except np.linalg.LinAlgError:
            idx = np.argmin(mu)
            
        return candidates[idx] * (ub - lb) + lb