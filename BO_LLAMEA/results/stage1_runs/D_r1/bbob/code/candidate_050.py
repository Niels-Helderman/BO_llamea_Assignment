import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5
        self.success_counter = 0
        self.failure_counter = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim, 10)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        def get_lcb(x, gp, beta=2.0):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Trust region constraints
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_lcb(x, self.gp), x0, bounds=list(zip(lb_tr, ub_tr)), 
                               method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None: next_x = np.random.uniform(lb_tr, ub_tr)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.success_counter += 1
                self.failure_counter = 0
                if self.success_counter >= 3:
                    self.length = min(1.0, self.length * 2.0)
                    self.success_counter = 0
            else:
                self.failure_counter += 1
                self.success_counter = 0
                if self.failure_counter >= 5:
                    self.length = max(0.01, self.length * 0.5)
                    self.failure_counter = 0
            
            if self.length < 0.02:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb