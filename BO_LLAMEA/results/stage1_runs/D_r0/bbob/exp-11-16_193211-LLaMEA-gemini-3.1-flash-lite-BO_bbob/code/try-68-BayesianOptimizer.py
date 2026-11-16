import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = max(2 * self.dim, 12)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_arr, y_arr = np.array(X_obs), np.array(y_obs)
            
            # Fit GP with periodic full hyperparameter optimization
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 15 == 0) else None
            try:
                self.gp.fit(X_arr, y_arr)
            except:
                pass
            
            # Acquisition: Log-EI (numerically stable improvement)
            def log_ei(x):
                x = x.reshape(1, -1)
                mu, std = self.gp.predict(x, return_std=True)
                std = np.maximum(std, 1e-12)
                diff = best_y - mu
                z = diff / std
                return -(diff * norm.cdf(z) + std * norm.pdf(z))

            # Optimize acquisition within trust region
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            next_x = None
            best_acq = np.inf
            # Multi-start optimization
            for x0 in [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(5)]:
                res = minimize(log_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None: next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(0.05, tr_size * 0.8)
                
        return best_y, best_x