import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Random Forest provides non-parametric uncertainty via variance of tree predictions
        self.model = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=1)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial sampling using Sobol
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Fit ensemble
            self.model.fit(X_data, y_data)
            
            # Acquisition: Expected Improvement (EI)
            # Use ensemble variance to estimate predictive uncertainty
            preds = np.array([tree.predict(X_data) for tree in self.model.estimators_])
            mu = np.mean(preds, axis=0)
            sigma = np.std(preds, axis=0) + 1e-9
            
            def ei(x):
                x = np.atleast_2d(x)
                mu_x = self.model.predict(x)
                # Compute std from ensemble dispersion
                std_x = np.std([tree.predict(x) for tree in self.model.estimators_], axis=0) + 1e-9
                
                diff = best_y - mu_x
                z = diff / std_x
                return -(diff * norm.cdf(z) + std_x * norm.pdf(z))
            
            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x