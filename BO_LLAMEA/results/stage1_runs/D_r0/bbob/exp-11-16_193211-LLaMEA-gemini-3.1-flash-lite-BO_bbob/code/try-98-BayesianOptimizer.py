import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1))
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-9, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        idx = np.argmin(y_obs)
        best_x = np.array(X_obs[idx])
        best_y = float(y_obs[idx])
        
        tr_size = 0.5
        tr_min = 0.01
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodic hyperparameter tuning
            self.gp.n_restarts_optimizer = 5 if i % 25 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            tr_lb = np.clip(best_x - tr_size * range_val, lb, ub)
            tr_ub = np.clip(best_x + tr_size * range_val, lb, ub)
            
            # Multi-start acquisition optimization
            best_next_val = float('inf')
            best_next_x = np.random.uniform(tr_lb, tr_ub)
            
            for _ in range(3):
                start_pt = np.random.uniform(tr_lb, tr_ub)
                res = minimize(ei, start_pt, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_next_val:
                    best_next_val, best_next_x = res.fun, res.x
            
            y_new = func(best_next_x)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), best_next_x
                tr_size = min(0.5, tr_size * 1.1)
            else:
                tr_size = max(tr_min, tr_size * 0.9)
                if tr_size <= tr_min:
                    tr_size = 0.5
            
            X_obs.append(best_next_x)
            y_obs.append(y_new)
                
        return best_y, best_x