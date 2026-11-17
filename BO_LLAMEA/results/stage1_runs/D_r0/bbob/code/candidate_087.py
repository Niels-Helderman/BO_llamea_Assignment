import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is robust for BBOB functions. 
        # WhiteKernel added to handle potential numerical instability during inversion.
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=2)
        self.tr_length = 0.5
        self.min_tr = 0.02
        self.max_tr = 1.0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = min(self.budget, 2 * self.dim + 4)
        
        # Sobol sequence for better space filling than pure random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
            if y < best_y:
                best_y, best_x = y, x
                # Reward successful steps by slightly expanding trust region
                self.tr_length = min(self.max_tr, self.tr_length * 1.1)
            else:
                # Shrink trust region on failure to focus exploration
                self.tr_length = max(self.min_tr, self.tr_length * 0.9)
            
        return best_y, best_x

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        y = np.array(y)
        self.gp.fit(X_norm, y)
        
        y_min = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        def acq(x_n):
            mu, sigma = self.gp.predict(x_n.reshape(1, -1), return_std=True)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / (sigma + 1e-12)
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # Minimize negative EI

        # Multistart search within the trust region + global perturbation
        candidates = [best_x_norm]
        for _ in range(8):
            candidates.append(np.clip(best_x_norm + np.random.normal(0, self.tr_length, self.dim), 0, 1))
        candidates.append(np.random.rand(self.dim))
        
        best_x_norm_prop = best_x_norm
        min_acq = np.inf
        
        for x0 in candidates:
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-5)
                if res.fun < min_acq:
                    min_acq, best_x_norm_prop = res.fun, res.x
            except: continue
            
        return np.clip(best_x_norm_prop, 0, 1) * (ub - lb) + lb