import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small noise floor to handle deterministic function precision
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # LHS-like initialization to ensure coverage
        n_init = max(2 * self.dim + 2, 8)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([float(func(x)) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def ucb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - beta * sigma

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Adaptive beta: reduce exploration as budget depletes
            progress = i / self.budget
            beta = 2.0 * (1.0 - progress) + 0.1
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start optimization of UCB
            candidates = np.random.uniform(lb, ub, (12, self.dim))
            candidates = np.vstack([candidates, best_x])
            
            for x0 in candidates:
                res = minimize(ucb, x0, args=(self.gp, beta), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback for numerical stability
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            try:
                val = float(func(next_x))
            except:
                val = np.inf
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = val
                best_x = next_x.copy()
                
        return best_y, best_x