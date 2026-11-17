import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X_obs, y_obs = [], []
        n_init = min(15, self.budget // 3)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = qmc.scale(sobol.random(n_init), lb, ub)
        
        best_y, best_x = np.inf, None

        for i in range(self.budget):
            if i < n_init:
                x_next = init_samples[i]
            else:
                if len(X_obs) % 5 == 0:
                    self.gp.fit(X_obs, y_obs)
                
                min_y = np.min(y_obs)
                def acq(x):
                    mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                    std = np.maximum(std, 1e-12)
                    imp = min_y - mu
                    z = imp / std
                    return -(imp * norm.cdf(z) + std * norm.pdf(z))

                best_acq = np.inf
                x_next = np.random.uniform(lb, ub)
                # Multi-start optimization
                for _ in range(10):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_next = res.fun, res.x
            
            y_next = float(func(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            
            if y_next < best_y:
                best_y, best_x = y_next, x_next
        
        return best_y, np.array(best_x)