import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel for deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for space-filling
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub)[0] for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        tr_min = 0.02
        
        for i in range(n_init, self.budget):
            # Periodically re-optimize hyperparameters to balance cost vs accuracy
            self.gp.n_restarts_optimizer = 2 if (i % 30 == 0) else 0
            try:
                self.gp.fit(np.array(X_obs), np.array(y_obs))
            except:
                pass
            
            # Log-Expected Improvement for exploitation
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # log(EI) = log(diff * Phi(z) + sigma * phi(z))
                return -(np.log(np.maximum(diff * norm.cdf(z) + sigma * norm.pdf(z), 1e-20)))

            # Constrain search to Trust Region around best_x
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            next_x = None
            best_acq = np.inf
            # Multi-start search
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(10)]
            for x0 in candidates:
                res = minimize(log_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                
        return best_y, best_x