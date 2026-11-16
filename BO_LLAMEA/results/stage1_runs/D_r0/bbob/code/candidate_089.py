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
        # Using a small alpha to ensure numerical stability during GP fitting
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-9, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initial sampling to better map the 6D landscape
        n_init = 4 * self.dim
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        idx = np.argmin(y_obs)
        best_x = np.array(X_obs[idx])
        best_y = float(y_obs[idx])
        
        # Hyperparameters for search strategy
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodic hyperparameter tuning to keep the model updated
            if i % 15 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -( (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

            # Alternate between local search (exploitation) and global random (exploration)
            if np.random.rand() < 0.8:
                tr_lb = np.clip(best_x - tr_size * (ub - lb), lb, ub)
                tr_ub = np.clip(best_x + tr_size * (ub - lb), lb, ub)
                res = minimize(ei, best_x, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                next_x = res.x if res.success else np.random.uniform(lb, ub)
            else:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(0.6, tr_size * 1.1)
            else:
                tr_size = max(0.1, tr_size * 0.95)
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x