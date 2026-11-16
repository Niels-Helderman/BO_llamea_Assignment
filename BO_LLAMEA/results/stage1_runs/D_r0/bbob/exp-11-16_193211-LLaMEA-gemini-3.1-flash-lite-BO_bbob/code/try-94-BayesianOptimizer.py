import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 2.5 is robust for BBOB functions; constant kernel for scale adaptation
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-8, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = np.array(X_obs[best_idx])
        best_y = float(y_obs[best_idx])
        
        tr_size = 0.5
        tr_min = 1e-3
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically re-optimize kernel to prevent overfitting
            self.gp.n_restarts_optimizer = 3 if i % 30 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Thompson Sampling: sample a function from the GP posterior
            mu, cov = self.gp.predict(X_data, return_cov=True)
            jitter = np.eye(len(X_data)) * 1e-6
            try:
                # Posterior sampling approximation
                weights = np.random.multivariate_normal(mu, cov + jitter)
            except:
                weights = mu
            
            def surrogate_objective(x):
                # Use GP prediction as surrogate
                pred, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                return pred[0] - 1.96 * std[0] # LCB for exploration

            # Multi-start acquisition optimization within trust region
            tr_lb = np.clip(best_x - tr_size * range_val, lb, ub)
            tr_ub = np.clip(best_x + tr_size * range_val, lb, ub)
            
            best_local_x = None
            best_local_val = np.inf
            
            for _ in range(4):
                start_x = np.random.uniform(tr_lb, tr_ub)
                res = minimize(surrogate_objective, start_x, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local_x = res.x
            
            next_x = best_local_x if best_local_x is not None else np.random.uniform(tr_lb, tr_ub)
            y_new = func(next_x)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(0.5, tr_size * 1.1)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                if tr_size <= tr_min:
                    tr_size = 0.5
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x