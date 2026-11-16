import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        n_init = 2 * self.dim + 2
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(np.array(x)) for x in X_obs]
        
        best_idx = np.argmin(y_obs)
        best_x = np.array(X_obs[best_idx])
        best_y = float(y_obs[best_idx])
        
        tr_size = 0.5
        tr_min = 0.05
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Refit GP every 10 steps to save cost, otherwise update only
            try:
                if i % 10 == 0:
                    self.gp.kernel.set_params(**self.gp.kernel.get_params())
                    self.gp.fit(X_data, y_data)
                else:
                    self.gp.fit(X_data, y_data)
            except:
                pass
            
            beta = 2.0 * (1.0 - (i - n_init) / (self.budget - n_init))
            
            # Trust region bounds
            tr_lb = np.maximum(lb, best_x - tr_size * (ub - lb))
            tr_ub = np.minimum(ub, best_x + tr_size * (ub - lb))
            
            def ucb(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                return (mu - beta * sigma)[0]

            # Optimize acquisition
            best_acq = np.inf
            next_x = None
            candidates = [best_x] + [np.random.uniform(tr_lb, tr_ub) for _ in range(8)]
            for x0 in candidates:
                res = minimize(ucb, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                tr_size = min(1.0, tr_size * 1.2)
            else:
                tr_size = max(tr_min, tr_size * 0.8)
                if tr_size <= tr_min:
                    tr_size = 0.5
                
        return best_y, best_x