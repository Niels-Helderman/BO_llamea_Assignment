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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = min(self.budget // 5, 20)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            # Fit model
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Acquisition: EI
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -((best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z))
            
            # Optimization of acquisition in Trust Region
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            res = minimize(ei, best_x_norm, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
            next_x = res.x if res.success else np.random.uniform(lb_tr, ub_tr)
            
            # Observe
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update TR
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = min(self.length * 1.2, 1.0)
            else:
                self.length *= 0.8
                if self.length < 1e-3:
                    self.length = 0.5
                    best_x_norm = X[np.argmin(y)]
                    
        return best_y, best_x_norm * (ub - lb) + lb