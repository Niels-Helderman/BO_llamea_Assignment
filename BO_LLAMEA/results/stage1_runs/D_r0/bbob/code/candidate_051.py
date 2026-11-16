import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # RandomForest is robust to high-dimensional input and non-stationary objective functions
        self.rf = RandomForestRegressor(n_estimators=100, min_samples_split=3, n_jobs=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # 1. Initialization using Sobol sequence for better space filling than random
        n_init = 2 * self.dim + 5
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub)
        y_obs = np.array([func(x) for x in X_obs])
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        # 2. Sequential Optimization
        for i in range(n_init, self.budget):
            # Standardize y for the forest
            y_mean, y_std = np.mean(y_obs), np.std(y_obs) + 1e-8
            y_norm = (y_obs - y_mean) / y_std
            
            self.rf.fit(X_obs, y_norm)
            
            # Acquisition: Expected Improvement over observed minimum
            def ei_acq(x):
                x = np.atleast_2d(x)
                preds = [tree.predict(x) for tree in self.rf.estimators_]
                mu = np.mean(preds, axis=0)
                sigma = np.std(preds, axis=0) + 1e-9
                
                y_best_norm = (best_y - y_mean) / y_std
                diff = y_best_norm - mu
                z = diff / sigma
                return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            # Candidates: best_x + random samples
            pool = np.vstack([best_x, qmc.scale(sampler.random(10), lb, ub)])
            for x0 in pool:
                res = minimize(ei_acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            # Evaluate objective
            y_new = func(next_x)
            X_obs = np.vstack([X_obs, next_x])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x