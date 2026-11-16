import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.length = 0.5
        # Use WhiteKernel for numerical stability with deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = min(self.budget, 2 * self.dim + 5)
        # Latin Hypercube Sampling approximation
        X = np.random.rand(n_init, self.dim)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y = float(y[best_idx])
        best_x_norm = X[best_idx].copy()
        
        success_counter = 0
        failure_counter = 0
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e12)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei

        for i in range(n_init, self.budget):
            # Periodically re-fit hyperparameters to ensure quality as data grows
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 1
            else:
                self.gp.n_restarts_optimizer = 0
                
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            lb_tr = np.clip(best_x_norm - self.length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length / 2, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start acquisition optimization
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_ei(x, self.gp, y_min), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None: 
                next_x = np.random.uniform(lb_tr, ub_tr)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    self.length = min(1.0, self.length * 1.2)
                    success_counter = 0
            else:
                failure_counter += 1
                if failure_counter >= 5:
                    self.length = max(0.01, self.length * 0.5)
                    failure_counter = 0
            
            if self.length < 0.05:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb