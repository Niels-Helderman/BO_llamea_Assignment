import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Enhanced initialization using Latin Hypercube Sampling
        n_init = min(30, self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_init)
        X_obs = qmc.scale(sample, lb, ub).tolist()
        y_obs = [func(np.array(x)) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_y = float(y_obs[best_idx])
        best_x = np.array(X_obs[best_idx])

        def get_ei(x, y_min):
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            with np.errstate(divide='ignore'):
                z = (y_min - mu) / sigma
                ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei[0]

        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit hyperparameters
            if (i - n_init) % 5 == 0:
                try:
                    self.gp.fit(X_data, y_data)
                except:
                    pass
            
            y_min = np.min(y_data)
            next_x = None
            best_acq = -np.inf
            
            # Multi-start acquisition optimization
            for _ in range(8):
                x0 = lb + (ub - lb) * np.random.rand(self.dim)
                res = minimize(get_ei, x0, args=(y_min,), bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
            
            try:
                y_new = func(next_x)
            except:
                next_x = lb + (ub - lb) * np.random.rand(self.dim)
                y_new = func(next_x)
                
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y = float(y_new)
                best_x = next_x
                
        return best_y, best_x