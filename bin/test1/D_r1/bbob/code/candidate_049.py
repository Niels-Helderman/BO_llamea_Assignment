import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for robust space-filling initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim, min(self.budget // 4, 15))
        X_raw = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_raw])
        X = X_raw
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def get_lcb(x, gp, beta=2.0):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                # Adaptive TR: constrain search space to trust region
                lb_tr = np.maximum(best_x - self.length, 0.0)
                ub_tr = np.minimum(best_x + self.length, 1.0)
                
                best_acq = np.inf
                next_x = None
                
                # Multi-start acquisition optimization
                for _ in range(8):
                    x0 = np.random.uniform(lb_tr, ub_tr)
                    res = minimize(lambda x: get_lcb(x, self.gp), x0, 
                                   bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x = res.x
            except:
                next_x = None
            
            # Fallback if optimization fails or GP is singular
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update TR and incumbent
            if val < best_y:
                best_y, best_x = float(val), next_x
                self.length = min(1.0, self.length * 1.2)
            else:
                self.length *= 0.8
                if self.length < 1e-3:
                    self.length = 0.5
        
        return best_y, best_x * (ub - lb) + lb