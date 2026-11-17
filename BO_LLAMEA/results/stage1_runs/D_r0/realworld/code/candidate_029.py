import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # Use Matérn 5/2 for smoothness assumptions; WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = np.zeros(self.dim)
        
        # Initial phase: Latin Hypercube-like spread using Halton/Sobol-inspired jittered grid
        n_init = max(2 * self.dim + 4, 10)
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * (np.random.rand(self.dim))
            else:
                # Update model less frequently as budget grows to save compute
                if i % 2 == 0 or len(self.X_hist) < 20:
                    try:
                        self.gp.fit(np.array(self.X_hist), np.array(self.y_hist))
                    except:
                        pass
                
                # Thompson Sampling approach: Sample from posterior and minimize
                # We perform local optimization of the posterior mean + noise perturbation
                def objective(x_norm):
                    x = x_norm * (ub - lb) + lb
                    mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                    # Add jitter for exploration
                    return float(mu + np.random.normal(0, 1) * std)

                # Multistart local optimization for the acquisition function
                best_acq = np.inf
                x_best = None
                for _ in range(4):
                    x0 = np.random.rand(self.dim)
                    res = minimize(objective, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                    if res.fun < best_acq:
                        best_acq = res.fun
                        x_best = res.x * (ub - lb) + lb
                x = np.clip(x_best, lb, ub)

            # Prevent sampling too close to existing points
            if len(self.X_hist) > 0:
                dist = np.linalg.norm((np.array(self.X_hist) - x) / (ub - lb), axis=1)
                if np.min(dist) < 1e-4:
                    x = lb + (ub - lb) * np.random.rand(self.dim)

            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
                    
        return float(best_y), np.array(best_x)