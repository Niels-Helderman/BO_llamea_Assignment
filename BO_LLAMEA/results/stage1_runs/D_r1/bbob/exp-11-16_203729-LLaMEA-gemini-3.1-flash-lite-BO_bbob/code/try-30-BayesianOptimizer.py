import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Random Forest as a robust, non-parametric surrogate for BBOB
        self.rf = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=1)
        self.length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        n_init = max(self.dim * 3, 12)
        X = np.random.uniform(0, 1, (n_init, self.dim))
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        success_counter = 0
        failure_counter = 0

        def get_ei(x, rf, y_min):
            # Forest variance estimate via standard deviation of tree predictions
            preds = np.array([tree.predict(x.reshape(1, -1)) for tree in rf.estimators_])
            mu = np.mean(preds)
            sigma = np.std(preds) + 1e-9
            diff = y_min - mu
            z = diff / sigma
            return diff * norm.cdf(z) + sigma * norm.pdf(z)

        for i in range(n_init, self.budget):
            try:
                self.rf.fit(X, (y - np.mean(y)) / (np.std(y) + 1e-9))
            except:
                pass
            
            y_min = np.min(y)
            lb_tr = np.clip(best_x_norm - self.length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length / 2, 0, 1)
            
            best_acq = -np.inf
            next_x = None
            
            # Local search for acquisition maximization
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: -get_ei(x, self.rf, y_min), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-3)
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    next_x = res.x
            
            if next_x is None: 
                next_x = np.random.uniform(lb_tr, ub_tr)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                success_counter += 1
                failure_counter = 0
                if success_counter >= 2:
                    self.length = min(1.0, self.length * 1.2)
                    success_counter = 0
            else:
                failure_counter += 1
                if failure_counter >= 3:
                    self.length = max(0.02, self.length * 0.6)
                    failure_counter = 0
            
            if self.length < 0.05:
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb