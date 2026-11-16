import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Use Sobol sequence for better space-filling initialization
        n_init = max(2 * self.dim + 2, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n=n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        success_counter, failure_counter = 0, 0

        def get_log_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e10)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / sigma
                log_ei = np.log(sigma + 1e-15) + norm.logcdf(z) + np.log(norm.pdf(z) + 1e-15)
            return -log_ei

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            lb_tr = np.clip(best_x_norm - self.length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length / 2, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start optimization with boundary enforcement
            for _ in range(10):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_log_ei(x, self.gp, y_min), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-5)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None: 
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                success_counter += 1
                failure_counter = 0
                if success_counter >= 2:
                    self.length = min(1.0, self.length * 1.1)
                    success_counter = 0
            else:
                failure_counter += 1
                if failure_counter >= 3:
                    self.length = max(0.02, self.length * 0.7)
                    failure_counter = 0
            
            # Occasional reset to global search if trapped
            if self.length < 0.03:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb