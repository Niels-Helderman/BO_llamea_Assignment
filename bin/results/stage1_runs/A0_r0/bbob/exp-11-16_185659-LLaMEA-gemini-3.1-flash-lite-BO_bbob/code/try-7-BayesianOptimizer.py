import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initialization
        n_init = max(2 * self.dim + 2, self.budget // 5)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n=n_init)
        X_obs = qmc.scale(sample, lb, ub).tolist()
        y_obs = [func(np.array(x)) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = np.array(X_obs[best_idx])
        best_y = float(y_obs[best_idx])
        
        for i in range(n_init, self.budget):
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Periodically re-optimize kernel hyperparameters
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i - n_init) % 8 == 0 else None
            try:
                self.gp.fit(X, y)
            except:
                pass
                
            y_min = np.min(y)
            
            def log_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = self.gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = y_min - mu
                z = diff / sigma
                # Log-EI is numerically more stable
                log_ei_val = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z) + np.log(1 + (sigma/np.maximum(diff, 1e-20)) * (norm.pdf(z)/norm.cdf(z)))
                return -log_ei_val
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = np.random.uniform(lb, ub)
            for _ in range(8):
                start_x = np.random.uniform(lb, ub)
                res = minimize(log_ei, start_x, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
                
        return best_y, best_x