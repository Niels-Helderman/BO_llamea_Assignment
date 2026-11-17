import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Using Random Forest as an ensemble surrogate to handle non-stationary objective functions
        self.model = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=1)
        self.tr_length = 0.5 
        self.min_tr = 0.05
        self.max_tr = 0.8

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        # Sufficient initial design for 6D
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                x = self._propose(np.array(X_obs), np.array(y_obs), lb, ub)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, X, y, lb, ub):
        X_norm = (X - lb) / (ub - lb)
        self.model.fit(X_norm, y)
        y_min = np.min(y)
        best_x_norm = X_norm[np.argmin(y)]
        
        # Acquisition: Expected Improvement using Forest variance as uncertainty estimate
        def acq(x_n):
            x_n = x_n.reshape(1, -1)
            preds = np.array([tree.predict(x_n)[0] for tree in self.model.estimators_])
            mu = np.mean(preds)
            sigma = np.std(preds) + 1e-9
            z = (y_min - mu) / sigma
            return -( (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

        best_acq_val = np.inf
        target_x = None
        
        # Multi-start optimization
        candidates = [best_x_norm + np.random.normal(0, self.tr_length, self.dim) for _ in range(15)]
        candidates.append(np.random.rand(self.dim))
        
        for x0 in candidates:
            try:
                res = minimize(acq, np.clip(x0, 0, 1), bounds=[(0, 1)]*self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    target_x = res.x
            except: continue
            
        if target_x is None:
            return np.random.uniform(lb, ub)
        return np.clip(target_x, 0, 1) * (ub - lb) + lb