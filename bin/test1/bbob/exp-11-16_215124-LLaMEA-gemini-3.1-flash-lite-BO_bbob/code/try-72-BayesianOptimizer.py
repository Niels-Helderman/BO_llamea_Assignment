import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel with small white noise for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Sobol sequence for high-quality space filling
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * scale
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(init_size, self.budget):
            # Update GP periodically; use jitter to prevent singularity
            if (i - init_size) % 10 == 0:
                self.gp.fit(X, y)
            
            # Beta schedule for UCB (starts high for exploration, lowers for exploitation)
            beta = 2.0 * np.log(i + 1)
            
            def ucb(x_norm):
                mu, sigma = self.gp.predict((lb + x_norm * scale).reshape(1, -1), return_std=True)
                return -(mu - beta * sigma)
            
            # Acquisition optimizer: Multi-start L-BFGS-B
            best_acq = -np.inf
            next_x_norm = np.random.rand(self.dim)
            for _ in range(5):
                x0 = np.random.rand(self.dim)
                res = minimize(ucb, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x_norm = res.x
            
            next_x = lb + next_x_norm * scale
            
            # Ensure unique points to avoid matrix issues
            if np.any(np.all(np.isclose(X, next_x), axis=1)):
                next_x = lb + np.random.rand(self.dim) * scale
                
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x