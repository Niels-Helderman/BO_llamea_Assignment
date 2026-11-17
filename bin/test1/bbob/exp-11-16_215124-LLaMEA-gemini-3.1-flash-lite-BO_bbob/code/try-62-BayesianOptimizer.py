import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel for BBOB: Matern 5/2 is standard; WhiteKernel handles deterministic jitter
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # Sobol sequence for better space-filling than LatinHypercube
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * scale
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        
        for i in range(init_size, self.budget):
            # Beta schedule for UCB: explore early, exploit late
            # Decreasing beta from 2.5 to 0.5 over the budget
            beta = 2.5 - 2.0 * (i / self.budget)
            
            try:
                # Refit kernel hyperparameters every 30 iterations to save compute
                self.gp.n_restarts_optimizer = 1 if (i % 30 == 0) else 0
                self.gp.fit(X, y)
                
                def objective(x_norm):
                    mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                    # LCB (Lower Confidence Bound) for minimization
                    return mu - beta * sigma

                # Multi-start acquisition optimization
                best_val = np.inf
                next_x_norm = np.random.rand(self.dim)
                for _ in range(10):
                    x0 = np.random.rand(self.dim)
                    res = minimize(objective, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * scale
            except:
                # Fallback to random search if GP fitting fails
                next_x = lb + np.random.rand(self.dim) * scale

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = float(val), next_x
                
        return best_y, best_x