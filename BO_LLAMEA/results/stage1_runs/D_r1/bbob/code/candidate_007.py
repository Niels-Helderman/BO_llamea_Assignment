import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is standard for BBOB; WhiteKernel handles numerical jitter
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # LHS initialization
        n_init = max(2 * self.dim + 2, 10)
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X[:, d] = np.linspace(lb[d], ub[d], n_init)
            np.random.shuffle(X[:, d])
        
        y = np.array([func(x) for x in X])
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        def lcb(x, gp, beta):
            mu, std = gp.predict(x.reshape(1, -1), return_std=True)
            return (mu - beta * std).item()

        for i in range(n_init, self.budget):
            # Fit GP, refit hyperparameters every 5 steps to save time
            if i % 5 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Adaptive beta schedule for UCB: starts high for exploration, decays
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            candidates = np.random.uniform(lb, ub, (8, self.dim))
            candidates = np.vstack([candidates, best_x])
            
            for x0 in candidates:
                res = minimize(lcb, x0, args=(self.gp, beta), 
                               bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            try:
                val = func(next_x)
            except:
                val = np.inf
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x