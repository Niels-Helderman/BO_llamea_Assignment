import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        self.rf = RandomForestRegressor(n_estimators=50, min_samples_split=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 4
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit models
            self.gp.fit(X_data, y_data)
            self.rf.fit(X_data, y_data)
            
            def ei(x):
                x = np.atleast_2d(x)
                mu_gp, std_gp = self.gp.predict(x, return_std=True)
                pred_rf = self.rf.predict(x)
                # Ensemble mean and variance (proxy)
                mu = 0.5 * (mu_gp + pred_rf)
                sigma = np.clip(std_gp, 1e-9, None)
                
                gamma = (best_y - mu) / sigma
                return -(sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma)))

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x