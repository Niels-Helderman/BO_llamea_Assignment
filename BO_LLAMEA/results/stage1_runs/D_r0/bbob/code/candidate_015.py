import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as a standard robust kernel for BBOB
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, alpha=1e-10, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design using Sobol sequence
        n_init = max(2 * self.dim + 2, min(self.budget // 4, 15))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                # Update GP periodically (every step, but cheap)
                X_norm = (np.array(X_obs) - lb) / (ub - lb)
                y_norm = np.array(y_obs)
                self.gp.fit(X_norm, y_norm)
                
                # Adaptive UCB: beta schedule to favor exploration early, exploitation late
                progress = i / self.budget
                beta = 2.0 * (1.0 - progress) + 0.1
                
                x = self._optimize_ucb(beta, lb, ub)
            
            y = func(x)
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _optimize_ucb(self, beta, lb, ub):
        def acq(x_norm):
            mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
            return (mu - beta * sigma).flatten()

        best_x = None
        min_acq = np.inf
        
        # Multi-start optimization of the acquisition function
        for _ in range(8):
            x0 = np.random.uniform(0, 1, self.dim)
            try:
                res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                if res.fun < min_acq:
                    min_acq = res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(best_x, 0, 1) * (ub - lb) + lb