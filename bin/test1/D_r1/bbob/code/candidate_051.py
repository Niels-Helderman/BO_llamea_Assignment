import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel ensemble for global and local regression
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + \
                 ConstantKernel(1.0) * RBF(length_scale=np.ones(dim)) + \
                 WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increased initial points to jumpstart convergence (min 2*dim)
        n_init = max(2 * self.dim + 2, min(self.budget // 3, 16))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_norm])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X_norm[best_idx].copy()
        
        def get_lcb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X_norm, y)
                # Adaptive beta for exploration-exploitation balance
                beta = 2.0 * (1 - (i / self.budget) ** 0.5) + 0.5
            except:
                beta = 0.0
            
            # Trust region boundaries
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start local optimization on acquisition function
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_lcb(x, self.gp, beta), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb_tr, ub_tr)
            
            val = func(next_x * (ub - lb) + lb)
            X_norm = np.vstack([X_norm, next_x])
            y = np.append(y, val)
            
            # Update best and trust-region parameters
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = min(1.0, self.length * 1.2)
            else:
                self.length = max(0.01, self.length * 0.8)
                
            # Periodic recovery from stagnation
            if i % 15 == 0 and self.length < 0.05:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb