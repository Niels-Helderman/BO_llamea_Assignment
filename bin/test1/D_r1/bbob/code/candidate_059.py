import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # ARD (Automatic Relevance Determination) enabled via individual length scales per dimension
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # LHS initialization for better coverage
        sampler = qmc.LatinHypercube(d=self.dim)
        n_init = max(2 * self.dim + 2, 10)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        # Log-transformation of y for target stabilization if y is strictly positive
        # Shift to ensure positivity
        y_min = np.min(y)
        y_shift = y_min - 0.1 if y_min <= 0 else 0
        y_trans = np.log(y - y_shift + 1e-9)
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Periodic refit with ARD optimizer restarts
                optimizer = 'fmin_l_bfgs_b' if (i % 10 == 0) else None
                self.gp.optimizer = optimizer
                self.gp.fit(X, y_trans)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                # Map back from log space
                mu_orig = np.exp(mu) + y_shift
                sigma_orig = sigma * np.exp(mu)
                sigma_orig = np.maximum(sigma_orig, 1e-12)
                
                z = (best_y - mu_orig) / sigma_orig
                return -( (best_y - mu_orig) * norm.cdf(z) + sigma_orig * norm.pdf(z) )
            
            # Multi-start acquisition optimization
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            for _ in range(3):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            y_trans = np.log(y - y_shift + 1e-9)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.2, 0.01, 1.0)
            else:
                self.length = np.clip(self.length * 0.8, 0.01, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb