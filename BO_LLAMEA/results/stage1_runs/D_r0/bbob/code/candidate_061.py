import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import qmc, norm

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Random Forest as the surrogate to avoid GP hyperparameter optimization costs and scaling issues
        self.model = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=1)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial design
        n_init = max(self.dim + 2, min(self.budget // 4, 15))
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
        # Normalize
        X_norm = (X - lb) / (ub - lb)
        self.model.fit(X_norm, y)
        
        y_min = np.min(y)
        
        # Random sampling for acquisition optimization
        n_candidates = 2000
        candidates = np.random.rand(n_candidates, self.dim)
        
        # Use Random Forest predictions + uncertainty (variance over trees)
        preds = [tree.predict(candidates) for tree in self.model.estimators_]
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0) + 1e-9
        
        # Expected Improvement calculation
        with np.errstate(divide='ignore'):
            imp = y_min - mu
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma <= 0] = 0
            
        # Select best candidate
        best_idx = np.argmax(ei)
        x_next = candidates[best_idx]
        
        return np.clip(x_next, 0, 1) * (ub - lb) + lb