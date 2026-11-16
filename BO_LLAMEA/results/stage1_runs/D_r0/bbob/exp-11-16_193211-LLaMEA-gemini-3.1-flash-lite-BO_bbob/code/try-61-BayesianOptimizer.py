import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space coverage in high dimensions
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs, y_obs = [], []
        for x in X_init:
            X_obs.append(x)
            y_obs.append(float(func(x)))
            
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit hyperparameters periodically to save compute
            self.gp.n_restarts_optimizer = 2 if i % 20 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Adaptive Exploration: beta decreases to focus on exploitation
            beta = 2.0 * (1.0 - (i / self.budget)**0.5)
            
            # Trust region centered at incumbent
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            def objective(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            starts = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(8)]
            for x0 in starts:
                res = minimize(objective, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback to random point if optimization failed
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = float(func(next_x))
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Update incumbent and adjust trust region
            if y_new < best_y:
                best_y, best_x = y_new, next_x
                tr_size = min(1.0, tr_size * 1.1)
            else:
                tr_size = max(0.01, tr_size * 0.9)
                
        return best_y, best_x