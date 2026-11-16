import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness assumptions; WhiteKernel prevents numerical instability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Dense Latin Hypercube Sampling for better initial coverage
        n_init = 20
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_init)
        X_obs = qmc.scale(sample, lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        def ei(x, gp, y_min):
            mu, sigma = gp.predict(np.atleast_2d(x), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            improve = y_min - mu
            z = improve / sigma
            return -(improve * norm.cdf(z) + sigma * norm.pdf(z))

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Re-fit hyperparameters every 5 steps to balance cost/performance
            if (i - n_init) % 5 == 0:
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            next_x = None
            best_acq = 0.0
            
            # Acquisition optimization: Multi-start L-BFGS-B
            candidates = [best_x] + [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(12)]
            for x0 in candidates:
                res = minimize(lambda x: ei(x, self.gp, min(y_obs)), x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Fallback if acquisition optimization fails or is non-informative
            if next_x is None or best_acq < 1e-9:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, np.array(best_x)