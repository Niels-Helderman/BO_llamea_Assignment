import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use ARD Matern kernel for better handling of different coordinate scales
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb
        
        # 1. Sobol-like deterministic space-filling design
        init_size = max(2 * self.dim + 4, 16)
        X = np.random.uniform(0, 1, (init_size, self.dim))
        y = np.array([func(lb + x * scale) for x in X])
        
        best_idx = np.argmin(y)
        best_x_norm, best_y = X[best_idx].copy(), y[best_idx]
        
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # 2. Local-biased fitting: focus on recent points to maintain local accuracy
                # Weights older points less or use sliding window if budget was larger
                self.gp.fit(X, y)
                
                # 3. Log-EI acquisition function for numerical stability
                def get_acquisition(x_norm):
                    x_norm = np.clip(x_norm, 0, 1)
                    mu, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    diff = best_y - mu
                    z = diff / sigma
                    # Log-EI is more robust to extreme values
                    return -(diff * norm.cdf(z) + sigma * norm.pdf(z))

                # 4. Multi-start optimization with adaptive trust region
                best_ei = -np.inf
                next_x_norm = None
                
                for _ in range(8):
                    x0 = np.clip(best_x_norm + np.random.normal(0, tr_len, self.dim), 0, 1)
                    res = minimize(get_acquisition, x0, bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                if next_x_norm is None: next_x_norm = np.random.uniform(0, 1, self.dim)
            except:
                next_x_norm = np.random.uniform(0, 1, self.dim)

            next_x = lb + next_x_norm * scale
            val = func(next_x)
            
            X = np.vstack([X, next_x_norm])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x_norm = val, next_x_norm
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.01, tr_len * 0.95)
                
        return float(best_y), lb + best_x_norm * scale