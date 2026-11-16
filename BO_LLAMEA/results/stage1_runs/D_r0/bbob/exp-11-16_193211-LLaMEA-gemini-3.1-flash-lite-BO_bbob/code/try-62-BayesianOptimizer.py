import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = max(12, 2 * self.dim + 4)
        
        sampler = qmc.LatinHypercube(d=self.dim)
        X_obs = list(qmc.scale(sampler.random(n_init), lb, ub))
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        tr_size = 0.5
        tr_min = 1e-4
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit hyperparameters every 20 steps to save compute
            self.gp.optimizer = 'fmin_l_bfgs_b' if (i % 20 == 0) else None
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # LCB with adaptive exploration beta
            beta = 2.0 * (1.0 - (i / self.budget)**0.5)
            
            # Shrinking trust region logic
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            def lcb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return mu - beta * sigma

            # Multi-start acquisition optimization within TR
            next_x = None
            best_acq = np.inf
            seeds = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(8)]
            for x0 in seeds:
                res = minimize(lcb, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
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
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size *= 0.8
                if tr_size < tr_min:
                    tr_size = 0.5 # Restart TR if collapsed
                
        return best_y, best_x