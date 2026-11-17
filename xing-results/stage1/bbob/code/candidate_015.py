import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        # Using a small fixed alpha for numerical stability in deterministic scenarios
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-9, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X_obs, y_obs = [], []
        # Increase initial points for better space coverage in 6D
        n_init = max(2 * self.dim + 1, min(self.budget // 4, 12))
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = sobol.random(n_init)
        
        best_y_curr, best_x_curr = np.inf, None

        for i in range(self.budget):
            if i < n_init:
                x_next = lb + (ub - lb) * init_samples[i]
            else:
                # Refit GP more frequently early on
                if i < 20 or (i - n_init) % 3 == 0:
                    try: self.gp.fit(np.array(X_obs), np.array(y_obs))
                    except: pass
                
                best_y = np.min(y_obs)
                def ei(x):
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                x_next = lb + (ub - lb) * np.random.rand(self.dim)
                best_acq = np.inf
                # Multi-start optimization with varied starting points
                for _ in range(8):
                    x0 = lb + (ub - lb) * np.random.rand(self.dim)
                    res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-4)
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            if y_next < best_y_curr:
                best_y_curr, best_x_curr = y_next, x_next
        
        return float(best_y_curr), np.array(best_x_curr)