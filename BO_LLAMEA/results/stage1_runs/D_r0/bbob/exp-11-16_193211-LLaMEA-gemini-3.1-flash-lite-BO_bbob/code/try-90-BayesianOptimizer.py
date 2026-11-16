import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Larger initial design for 6D space, better coverage
        n_init = 2 * self.dim + 4 
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = np.array(X_obs[best_idx])
        best_y = float(y_obs[best_idx])
        
        tr_size = 0.4
        tr_min = 1e-3
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodical hyperparameter tuning to save computation
            self.gp.n_restarts_optimizer = 3 if (i % 15 == 0) else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -( (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

            # Multi-start acquisition optimization to avoid local optima
            tr_lb = np.clip(best_x - tr_size * (ub - lb), lb, ub)
            tr_ub = np.clip(best_x + tr_size * (ub - lb), lb, ub)
            
            res = minimize(ei, best_x, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
            
            if res.success:
                next_x = res.x
            else:
                next_x = np.random.uniform(tr_lb, tr_ub)
                
            y_new = func(next_x)
            
            # Update best found
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(0.6, tr_size * 1.1)
            else:
                tr_size = max(tr_min, tr_size * 0.9)
                # Reset strategy: if trust region is too small, jump to explore
                if tr_size <= tr_min:
                    tr_size = 0.5 
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x