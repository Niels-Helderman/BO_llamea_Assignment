import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        self.rf = RandomForestRegressor(n_estimators=50, min_samples_split=3)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # LHS-like initialization
        n_init = max(2 * self.dim + 2, 12)
        X = np.random.uniform(lb, ub, (n_init, self.dim))
        y = np.array([func(x) for x in X])
        
        best_y, best_x = float(np.min(y)), X[np.argmin(y)].copy()

        def acquisition(x, y_min):
            x_reshaped = x.reshape(1, -1)
            # Ensemble: Average prediction from GP and RF
            mu_gp, std_gp = self.gp.predict(x_reshaped, return_std=True)
            mu_rf = self.rf.predict(x_reshaped)
            mu = 0.7 * mu_gp + 0.3 * mu_rf
            
            sigma = np.clip(std_gp, 1e-12, None)
            z = (y_min - mu) / sigma
            return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))

        for i in range(n_init, self.budget):
            # Fit surrogates
            try:
                self.gp.fit(X, y)
                self.rf.fit(X, y)
            except:
                pass
            
            y_min = np.min(y)
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            candidates = np.random.uniform(lb, ub, (12, self.dim))
            candidates = np.vstack([candidates, best_x])
            
            for x0 in candidates:
                res = minimize(acquisition, x0, args=(y_min,), bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub, self.dim)
            
            try:
                val = func(next_x)
            except:
                val = np.inf
                
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x = next_x.copy()
                
        return best_y, best_x