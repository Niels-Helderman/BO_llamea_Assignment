import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 2.5 is robust for BBOB functions; include WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 5
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = [qmc.scale(sampler.random(1), lb, ub).flatten() for _ in range(n_init)]
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Periodically refit hyperparameters for long-term robustness
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 1
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Beta schedule for UCB
            beta = 2.0 * np.sqrt(np.log(i + 1))
            
            # Constrain search to Trust Region
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return (mu - beta * sigma).item()

            # Acquisition optimization using multiple starts
            next_x = None
            best_acq = np.inf
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(10)]
            
            for x0 in candidates:
                res = minimize(ucb, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            # Update best and dynamic Trust Region
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.1)
            else:
                tr_size = max(0.01, tr_size * 0.95)
                # Escape local optima
                if tr_size < 0.05:
                    tr_size = 0.5
                
        return best_y, best_x