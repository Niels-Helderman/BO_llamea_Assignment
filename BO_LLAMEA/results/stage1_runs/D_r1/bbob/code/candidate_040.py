import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initialization
        sampler = Sobol(d=self.dim, scramble=True)
        n_init = min(self.budget // 5, 2 * self.dim + 2)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            if i % 5 == 0 or i == n_init:
                self.gp.fit(X, y)
            
            # Adaptive UCB beta: starts high, decays to prioritize exploitation as budget nears end
            t = (i + 1)
            beta = 2.0 * np.sqrt(np.log(t + 1))
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma
            
            # Multi-start optimization of the acquisition function
            best_acq = np.inf
            next_x = np.random.uniform(0, 1, self.dim)
            
            for _ in range(8):
                x0 = np.random.uniform(0, 1, self.dim)
                res = minimize(ucb, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            val = func(next_x * (ub - lb) + lb)
            
            # Update data
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
            
        return best_y, best_x_norm * (ub - lb) + lb