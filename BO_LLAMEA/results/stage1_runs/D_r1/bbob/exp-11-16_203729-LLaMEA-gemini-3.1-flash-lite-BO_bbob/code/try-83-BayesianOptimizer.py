import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel with fixed jitter to handle deterministic nature
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0, 
            normalize_y=True, 
            n_restarts_optimizer=2
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling
        n_init = min(self.budget, max(2 * self.dim + 2, 12))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Sequential optimization
        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
                
                # Adaptive UCB: beta decreases over iterations to transition from exploration to exploitation
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                def acq(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    # UCB for minimization: mu - beta * sigma
                    return mu - beta * sigma

                # Multi-start acquisition optimization
                best_acq_val = np.inf
                next_x = np.random.rand(self.dim)
                
                starts = [best_x_norm] + [np.random.rand(self.dim) for _ in range(4)]
                for x0 in starts:
                    res = minimize(acq, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        next_x = res.x
            except:
                # Fallback to random if GP fitting or optimization fails
                next_x = np.random.rand(self.dim)
            
            # Clamp to domain and evaluate
            next_x = np.clip(next_x, 0, 1)
            val = func(next_x * (ub - lb) + lb)
            
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                
        return best_y, best_x_norm * (ub - lb) + lb