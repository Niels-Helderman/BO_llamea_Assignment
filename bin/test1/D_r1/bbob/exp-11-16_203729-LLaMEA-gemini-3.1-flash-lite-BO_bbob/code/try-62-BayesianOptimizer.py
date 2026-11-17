import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is excellent for smooth but non-differentiable BBOB functions
        # Adding a small WhiteKernel to handle potential noise/numerical jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling than random
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 4, 16)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Periodic refit to optimize hyperparameters
                self.gp.n_restarts_optimizer = 1 if i % 5 == 0 else 0
                self.gp.fit(X, y)
            except:
                pass
            
            # Log-Expected Improvement acquisition function for better numerical stability
            def neg_log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # log(EI) = log(diff * Phi(z) + sigma * phi(z))
                # Computed via logaddexp for stability
                log_ei = np.logaddexp(np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z), 
                                      np.log(sigma) + norm.logpdf(z))
                return -log_ei
            
            # Adaptive Trust Region search
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            for _ in range(4):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(neg_log_ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Success-based trust region adaptation
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.05, 1.0)
            else:
                self.length = np.clip(self.length * 0.9, 0.05, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb