import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is robust for BBOB functions, includes WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def _get_ei(self, x, gp, y_best):
        x = x.reshape(1, -1)
        mu, std = gp.predict(x, return_std=True)
        with np.errstate(divide='ignore'):
            imp = y_best - mu
            z = imp / (std + 1e-9)
            ei = imp * norm.cdf(z) + std * norm.pdf(z)
            ei[std <= 0] = 0.0
        return -ei[0]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X, y = [], []
        
        # Initial points: use a small Sobol-like or Latin Hypercube approach (random here for simplicity)
        n_init = min(self.budget // 5, 2 * self.dim + 2)
        
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < n_init:
                x = np.random.uniform(lb, ub)
            else:
                # Optimize Acquisition Function
                y_best = np.min(y)
                best_acq = np.inf
                x_next = None
                
                # Multi-start gradient-based optimization of EI
                for _ in range(10):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(self._get_ei, x0, args=(self.gp, y_best), 
                                   bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_next = res.x
                x = x_next

            # Evaluate
            val = func(x)
            X.append(x)
            y.append(val)
            
            if val < best_y:
                best_y = val
                best_x = x

            # Refit GP periodically or as data grows
            if len(X) >= n_init:
                # Limit hyperparameter tuning frequency to save budget/time
                optimizer = 'fmin_l_bfgs_b' if (len(X) % 5 == 0 or len(X) == n_init) else None
                self.gp.optimizer = optimizer
                self.gp.fit(np.array(X), np.array(y))
        
        return float(best_y), best_x