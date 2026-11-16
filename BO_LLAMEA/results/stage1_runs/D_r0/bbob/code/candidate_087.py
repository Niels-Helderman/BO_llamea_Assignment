import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 2.5 is robust for BBOB; ConstantKernel handles scale
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        # Using a small alpha for numerical stability in deterministic scenarios
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=1e-10, 
            normalize_y=True, 
            n_restarts_optimizer=0
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial design
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init = qmc.scale(sampler.random(n_init), lb, ub)
        
        X_obs = [x for x in X_init]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit kernel hyperparameters every 15 steps to balance cost and accuracy
            self.gp.n_restarts_optimizer = 3 if i % 15 == 0 else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Acquisition: Expected Improvement
            def neg_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Multi-start optimization of EI
            # Combine current best with random points to ensure exploration/exploitation balance
            candidates = [best_x] + [np.random.uniform(lb, ub) for _ in range(4)]
            next_x = None
            best_acq = np.inf
            
            for x0 in candidates:
                res = minimize(neg_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B', tol=1e-5)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if acquisition optimization fails or provides a duplicate
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
                
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, best_x