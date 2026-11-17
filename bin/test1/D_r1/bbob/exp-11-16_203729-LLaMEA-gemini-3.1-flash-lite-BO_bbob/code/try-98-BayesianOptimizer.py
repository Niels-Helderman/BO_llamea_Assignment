import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.tr_size = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(max(2 * self.dim + 2, 10), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_norm])
        
        best_y = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        for i in range(n_init, self.budget):
            try:
                # Refit hyperparameters every 20 iterations to save cost
                if (i - n_init) % 20 == 0:
                    self.gp.optimizer = 'fmin_l_bfgs_b'
                else:
                    self.gp.optimizer = None
                self.gp.fit(X_norm, y)
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    z = (best_y - mu) / sigma
                    return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))

                # Adaptive Trust-Region search
                best_acq = np.inf
                next_x = None
                
                # Sample candidates in trust region and global space
                centers = [best_x_norm, np.random.rand(self.dim)]
                for c in centers:
                    bounds = [(max(0, c[j]-self.tr_size), min(1, c[j]+self.tr_size)) for j in range(self.dim)]
                    res = minimize(acq, c, bounds=bounds, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, next_x = res.fun, res.x
            except:
                next_x = np.random.rand(self.dim)
            
            if next_x is None:
                next_x = np.random.rand(self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X_norm = np.vstack([X_norm, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.tr_size = min(self.tr_size * 1.1, 1.0)
            else:
                self.tr_size = max(self.tr_size * 0.8, 0.05)
                if self.tr_size <= 0.06: self.tr_size = 0.5 # Reset TR if too small
                
        return float(best_y), best_x_norm * (ub - lb) + lb