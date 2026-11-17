import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel ensemble with noise floor
        kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + 
                  ConstantKernel(1.0) * RBF(length_scale=np.ones(dim))) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial sampling for better global coverage
        n_init = max(2 * self.dim + 2, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        def get_ucb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            # Fit GP with adaptive regularization
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Dynamic beta: aggressive early exploration, tighter late exploitation
            beta = 2.0 * (1 - (i / self.budget)**0.5)
            
            # Trust region boundaries
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start acquisition optimization
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_ucb(x, self.gp, beta), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback to random uniform search
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update best and dynamic length scaling
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = min(1.0, self.length * 1.2)
            else:
                self.length = max(0.05, self.length * 0.9)
            
            # Reset length if local search stagnates
            if i % 15 == 0 and self.length < 0.1:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb