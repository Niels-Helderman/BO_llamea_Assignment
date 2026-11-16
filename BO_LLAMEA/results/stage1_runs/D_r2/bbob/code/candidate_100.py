import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-9)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # Sobol sequence for better space filling in early iterations
        init_size = min(max(2 * self.dim + 2, 20), self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters every 20 steps to balance overhead/accuracy
                if i % 20 == 0:
                    self.gp.n_restarts_optimizer = 2
                else:
                    self.gp.n_restarts_optimizer = 0
                self.gp.fit(X, y)
                
                def get_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    from scipy.stats import norm
                    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei

                best_ei = -np.inf
                next_x_norm = np.random.rand(self.dim)
                
                # Multi-start acquisition optimization in an adaptive trust region
                tr_center = (best_x - lb) / range_val
                for _ in range(12):
                    tr_lb = np.clip(tr_center - tr_len, 0, 1)
                    tr_ub = np.clip(tr_center + tr_len, 0, 1)
                    x0 = np.clip(tr_center + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                # Fallback to random sampling if GP fitting or optimization fails
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update best and manage trust region size
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.01, tr_len * 0.95)
                
        return best_y, best_x