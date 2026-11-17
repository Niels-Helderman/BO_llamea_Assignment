import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc, norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 for smoothness and WhiteKernel for numerical stability
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        # LHS initialization for robust space coverage
        init_size = min(max(2 * self.dim + 2, 16), self.budget // 4)
        sampler = qmc.LatinHypercube(d=self.dim)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), float(y[best_idx])
        tr_len = 0.4
        
        for i in range(init_size, self.budget):
            # Periodically re-optimize hyperparameters to handle cumulative data
            self.gp.n_restarts_optimizer = 2 if (i % 50 == 0) else 0
            try:
                self.gp.fit(X, y)
                
                def get_neg_ei(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Multi-start acquisition within adaptive trust region
                tr_center = (best_x - lb) / range_val
                best_ei = -np.inf
                next_x_norm = np.clip(tr_center + np.random.normal(0, tr_len/2, self.dim), 0, 1)
                
                for _ in range(10):
                    x0 = np.clip(tr_center + np.random.normal(0, tr_len, self.dim), 0, 1)
                    res = minimize(get_neg_ei, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei, next_x_norm = -res.fun, res.x
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update trust region based on success
            if val < best_y:
                best_y, best_x = float(val), next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len *= 0.95
                if tr_len < 0.05: tr_len = 0.5
        
        return best_y, best_x