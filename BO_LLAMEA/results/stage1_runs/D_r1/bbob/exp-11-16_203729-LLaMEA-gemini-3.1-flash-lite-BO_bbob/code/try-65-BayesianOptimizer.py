import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, 12)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Beta schedule for UCB (start high, cool down for exploration-exploitation balance)
        beta_start = 2.0
        beta_end = 0.2
        
        for i in range(n_init, self.budget):
            try:
                # Fit model; optimize hyperparams rarely to keep cost down
                self.gp.n_restarts_optimizer = 1 if (i % 20 == 0) else 0
                self.gp.fit(X, y)
            except:
                pass
            
            beta = beta_start - (i / self.budget) * (beta_start - beta_end)
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Trust region boundaries
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            for _ in range(4):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(ucb, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.9, 0.05, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb