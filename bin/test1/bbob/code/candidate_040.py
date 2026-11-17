import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

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
        
        # Sobol sequence for space-filling initialization
        init_size = min(max(10, self.dim * 2), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = lb + sampler.random(n=init_size) * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters only periodically
                self.gp.n_restarts_optimizer = 1 if i % 20 == 0 else 0
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Adaptive search: mix local trust region with occasional global random probe
                best_ei = -np.inf
                next_x_norm = None
                tr_center = (best_x - lb) / range_val
                
                # Perform multi-start optimization within trust region
                for _ in range(5):
                    tr_lb = np.clip(tr_center - tr_len, 0, 1)
                    tr_ub = np.clip(tr_center + tr_len, 0, 1)
                    x0 = np.clip(np.random.normal(tr_center, tr_len / 3.0), 0, 1)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei, next_x_norm = -res.fun, res.x
                
                # Fallback to random if acquisition optimization fails or is too flat
                if next_x_norm is None or best_ei < 1e-9:
                    next_x = lb + np.random.rand(self.dim) * range_val
                else:
                    next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Trust region update logic
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len *= 0.9
                if tr_len < 0.01:
                    tr_len = 0.5
        
        return float(best_y), best_x