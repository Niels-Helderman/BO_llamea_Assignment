import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial design size for better early-stage exploration
        n_init = min(self.budget // 3, 20 + 2 * self.dim)
        sampler = qmc.LatinHypercube(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Only optimize hyperparameters periodically to save compute
            optimize = (i - n_init) % 5 == 0
            self.gp.n_restarts_optimizer = 3 if optimize else 0
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Trust region boundaries
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            # Multi-start optimization with increased candidates
            for _ in range(6):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(log_ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.2, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.8, 0.05, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb