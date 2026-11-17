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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Latin Hypercube Sampling for better coverage
        sampler = qmc.LatinHypercube(d=self.dim)
        n_init = min(self.budget // 3, 20)
        X = sampler.random(n_init)
        y = np.array([func(x * scale + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Fit GP with minimal restarts to keep overhead low
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Log-Expected Improvement for better numerical stability in exploitation
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI: log(diff * Phi(z) + sigma * phi(z))
                # Using a safe implementation
                log_ei_val = np.log(np.maximum(diff * norm.cdf(z) + sigma * norm.pdf(z), 1e-300))
                return -log_ei_val
            
            # Trust region boundaries
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            # Multi-start optimization within trust region
            for _ in range(5):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(log_ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * scale + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.2, 0.01, 1.0)
            else:
                self.length = np.clip(self.length * 0.8, 0.01, 1.0)
                
        return best_y, best_x_norm * scale + lb