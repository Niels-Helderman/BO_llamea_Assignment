import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use an ensemble of 50 trees to estimate uncertainty via variance
        self.rf = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space filling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = 2 * self.dim + 2
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit surrogate
            self.rf.fit(X_data, y_data)
            
            # Expected Improvement Acquisition
            def acquisition(x):
                x = np.atleast_2d(x)
                preds = np.array([tree.predict(x)[0] for tree in self.rf.estimators_])
                mu = np.mean(preds)
                sigma = np.std(preds) + 1e-9
                
                diff = best_y - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # Minimize negative EI
            
            # Optimize acquisition with multi-start
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            best_acq = np.inf
            next_x = None
            
            for x0 in candidates:
                res = minimize(acquisition, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x