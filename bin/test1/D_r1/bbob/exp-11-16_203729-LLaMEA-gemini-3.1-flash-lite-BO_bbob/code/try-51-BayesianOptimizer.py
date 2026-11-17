import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * (Matern(nu=2.5, length_scale_bounds='fixed') + RBF(length_scale_bounds='fixed')) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialization
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_raw = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X_raw])
        X = X_raw
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx]
        
        def get_ucb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            # Refit GP
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Adaptive beta: cooling schedule (exploration to exploitation)
            progress = (i - n_init) / (self.budget - n_init)
            beta = 2.0 * (1.0 - progress) + 0.5
            
            # Optimization of acquisition function
            best_acq = np.inf
            next_x = np.random.rand(self.dim)
            
            # Multi-start gradient-based optimization
            starts = np.vstack([best_x, np.random.rand(5, self.dim)])
            for x0 in starts:
                res = minimize(lambda x: get_ucb(x, self.gp, beta), x0, 
                               bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization failed
            val = func(next_x * (ub - lb) + lb)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x * (ub - lb) + lb