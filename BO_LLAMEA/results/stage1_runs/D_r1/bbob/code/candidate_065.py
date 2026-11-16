import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.6

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling is often more robust for very low n_init than pure Sobol
        sampler = qmc.LatinHypercube(d=self.dim, scramble=True)
        n_init = min(self.budget, max(12, 2 * self.dim + 2))
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Refit hyperparameters only every few steps to maintain speed
                self.gp.n_restarts_optimizer = 1 if i % 10 == 0 else 0
                self.gp.fit(X, y)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Focused local search centered on best_x
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            # Multi-start optimization inside the trust region
            for _ in range(6):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Global fallback to avoid getting trapped in local regions
            if next_x is None or np.random.rand() < 0.1:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Trust region logic: tighten on failure, loosen on success
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.2, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.8, 0.05, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb