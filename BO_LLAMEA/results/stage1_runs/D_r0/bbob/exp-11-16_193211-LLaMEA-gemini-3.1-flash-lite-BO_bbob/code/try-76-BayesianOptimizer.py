import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub)[0] for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x, best_y = X_obs[best_idx], float(y_obs[best_idx])
        
        tr_size = 0.5
        for i in range(n_init, self.budget):
            X_data, y_data = np.array(X_obs), np.array(y_obs)
            
            # Periodically re-fit hyperparams, use more restarts for initial data
            self.gp.n_restarts_optimizer = 3 if i < 50 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Thompson Sampling via posterior sampling
            # Approximate the GP path at a set of candidate points
            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            def objective(x):
                # Sample from posterior
                mu, cov = self.gp.predict(x.reshape(1, -1), return_cov=True)
                sample = np.random.multivariate_normal(mu.flatten(), cov + 1e-10 * np.eye(1))
                return sample

            next_x = None
            best_acq = np.inf
            # Multi-start optimization of the random sample
            for _ in range(5):
                x0 = np.random.uniform(tr_lb, tr_ub)
                res = minimize(objective, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(0.02, tr_size * 0.95)
                
        return best_y, best_x