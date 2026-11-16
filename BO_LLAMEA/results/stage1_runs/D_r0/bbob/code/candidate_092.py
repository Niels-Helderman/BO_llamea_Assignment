import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
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
        
        # Initialization
        n_init = 2 * self.dim + 2
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = np.array(X_obs[np.argmin(y_obs)])
        best_y = float(np.min(y_obs))
        
        # Optimization Loop
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically update hyperparams to save compute
            self.gp.n_restarts_optimizer = 5 if (i % 25 == 0) else 0
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass

            # EI acquisition function
            def ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                z = (best_y - mu) / sigma
                return -( (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z) )

            # Multi-start acquisition optimization
            best_acq = np.inf
            next_x = None
            
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(3)]
            for start_point in starts:
                res = minimize(ei, start_point, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            # Fallback if optimization fails
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            # Evaluate objective
            y_new = func(next_x)
            
            # Update best
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
            
            X_obs.append(next_x)
            y_obs.append(y_new)
                
        return best_y, best_x