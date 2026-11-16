import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Use Sobol sequence for better space filling coverage than random LHC
        n_init = min(self.budget // 4, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n=n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.clip(sigma, 1e-12, 1e6)
            diff = y_min - mu
            z = diff / sigma
            return diff * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            lb_tr = np.clip(best_x_norm - self.length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length / 2, 0, 1)
            
            # Multi-start optimization with a focus on trust region boundary
            best_acq = -np.inf
            next_x = None
            
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: -get_ei(x, self.gp, y_min), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-4)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            if next_x is None: 
                next_x = np.random.uniform(lb_tr, ub_tr)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                self.length = min(1.0, self.length * 1.2)
            else:
                self.length = max(0.02, self.length * 0.8)
                
        return best_y, best_x_norm * (ub - lb) + lb