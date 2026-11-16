import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        range_val = ub - lb
        
        init_size = min(max(2 * self.dim + 2, 10), self.budget // 4)
        X, y = [], []
        for _ in range(init_size):
            x = np.random.uniform(lb, ub)
            X.append(x)
            y.append(func(x))
        
        X, y = np.array(X), np.array(y)
        best_y, best_x = np.min(y), X[np.argmin(y)].copy()
        
        tr_len = 0.5
        for i in range(init_size, self.budget):
            try:
                self.gp.fit(X, y)
                # Adaptive beta schedule for UCB
                beta = 2.0 * np.log(i + 1)
                
                def get_ucb(x_norm):
                    x = lb + x_norm * range_val
                    mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                    return mu - beta * sigma
                
                best_val = np.inf
                next_x = None
                tr_center = (best_x - lb) / range_val
                
                for _ in range(8):
                    tr_lb = np.clip(tr_center - tr_len, 0, 1)
                    tr_ub = np.clip(tr_center + tr_len, 0, 1)
                    x0 = np.random.uniform(tr_lb, tr_ub)
                    res = minimize(get_ucb, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B')
                    if res.fun < best_val:
                        best_val = res.fun
                        next_x = lb + res.x * range_val
                
                if next_x is None: next_x = np.random.uniform(lb, ub)
            except:
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len = max(0.05, tr_len * 0.95)
                
        return float(best_y), best_x