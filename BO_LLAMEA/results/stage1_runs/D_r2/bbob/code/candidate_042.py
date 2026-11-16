import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm, qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Use an ensemble of GPs with different length scales to balance global/local search
        self.kernels = [
            ConstantKernel(1.0) * Matern(length_scale=np.ones(dim) * 0.5, nu=2.5) + WhiteKernel(1e-10),
            ConstantKernel(1.0) * Matern(length_scale=np.ones(dim) * 0.2, nu=1.5) + WhiteKernel(1e-10)
        ]
        self.models = [GaussianProcessRegressor(kernel=k, alpha=1e-10, normalize_y=True) for k in self.kernels]
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        init_size = min(self.budget // 6, 40)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_norm = sampler.random(n=init_size)
        X = lb + X_norm * range_val
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        tr_len = 0.5
        
        for i in range(init_size, self.budget):
            try:
                # Fit ensemble
                for model in self.models:
                    model.fit(X, y)
                
                def get_ei(x_norm):
                    x = (lb + x_norm * range_val).reshape(1, -1)
                    preds = [m.predict(x, return_std=True) for m in self.models]
                    mus = np.array([p[0] for p in preds])
                    sigmas = np.array([p[1] for p in preds])
                    # Aggregate using Mean Prediction + Max Uncertainty (conservative ensemble)
                    mu = np.mean(mus)
                    sigma = np.sqrt(np.mean(sigmas**2 + mus**2) - mu**2 + np.mean(sigmas**2))
                    sigma = np.maximum(sigma, 1e-12)
                    imp = best_y - mu
                    Z = imp / sigma
                    return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                best_ei = -np.inf
                next_x_norm = None
                tr_center = (best_x - lb) / range_val
                
                # Multi-start local optimization within trust region
                for _ in range(8):
                    tr_lb = np.clip(tr_center - tr_len, 0, 1)
                    tr_ub = np.clip(tr_center + tr_len, 0, 1)
                    x0 = np.clip(np.random.normal(tr_center, tr_len / 2.0), 0, 1)
                    res = minimize(get_ei, x0, bounds=list(zip(tr_lb, tr_ub)), method='L-BFGS-B', tol=1e-4)
                    if -res.fun > best_ei:
                        best_ei = -res.fun
                        next_x_norm = res.x
                
                next_x = lb + next_x_norm * range_val
            except:
                next_x = lb + np.random.rand(self.dim) * range_val

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                tr_len = min(1.0, tr_len * 1.1)
            else:
                tr_len *= 0.9
                if tr_len < 0.02: tr_len = 0.5
                
        return float(best_y), best_x