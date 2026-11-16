import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel ensemble for global/local mapping
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, self.budget // 5)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_y = np.min(y)
        best_x = X[np.argmin(y)].copy()

        def get_ei(x, gp, y_best):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            imp = y_best - mu
            z = imp / sigma
            return (imp * norm.cdf(z) + sigma * norm.pdf(z)).flatten()

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            # Multi-start acquisition optimization
            best_acq = -np.inf
            next_x = np.random.rand(self.dim)
            
            for _ in range(8):
                x0 = np.random.rand(self.dim)
                res = minimize(lambda x: -get_ei(x, self.gp, best_y), x0, 
                               bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = val
                best_x = next_x
            
            # Periodic refitting optimization
            if i % 50 == 0:
                self.gp.kernel_.set_params(**self.gp.kernel_.get_params())
                
        return float(best_y), best_x * (ub - lb) + lb