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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_data = qmc.scale(sampler.random(n_init), lb, ub)
        y_data = np.array([func(x) for x in X_data])
        
        best_idx = np.argmin(y_data)
        best_x = X_data[best_idx].copy()
        best_y = float(y_data[best_idx])
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            if i % 20 == 0:
                self.gp.n_restarts_optimizer = 3
                self.gp.fit(X_data, y_data)
                self.gp.n_restarts_optimizer = 0
            else:
                self.gp.fit(X_data, y_data)
            
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                imp = best_y - mu
                Z = imp / sigma
                log_ei_val = np.log(np.maximum(imp * norm.cdf(Z) + sigma * norm.pdf(Z), 1e-30))
                return -log_ei_val # Minimize negative

            tr_lb = np.maximum(lb, best_x - tr_size * range_val)
            tr_ub = np.minimum(ub, best_x + tr_size * range_val)
            
            res = minimize(log_ei, best_x, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
            next_x = res.x if res.success else np.random.uniform(tr_lb, tr_ub)
            
            y_new = func(next_x)
            X_data = np.vstack([X_data, next_x])
            y_data = np.append(y_data, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size *= 0.8
                if tr_size < 1e-3:
                    tr_size = 0.5 # Restart TR
            
        return best_y, best_x