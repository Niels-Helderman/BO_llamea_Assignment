import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        init_size = max(2 * self.dim + 4, 16)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                tr_center = (best_x - lb) / range_val
                tr_lb = np.clip(tr_center - tr_len, 0, 1)
                tr_ub = np.clip(tr_center + tr_len, 0, 1)
                
                # Hybrid acquisition optimization: DE for global search, L-BFGS-B for local refinement
                if i % 10 == 0:
                    res = differential_evolution(get_ei, bounds=list(zip([0]*self.dim, [1]*self.dim)), maxiter=20, popsize=10)
                    next_x_norm = res.x
                else:
                    res = minimize(get_ei, (tr_lb + tr_ub)/2, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    next_x_norm = res.x
                
                next_x = np.clip(lb + next_x_norm * range_val, lb, ub)
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.05, tr_len * 0.9)
                
        return float(best_y), best_x