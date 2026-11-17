import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Matern 5/2 is standard for BBOB; WhiteKernel adds robustness to numerical noise.
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=True)
        self.min_y = np.inf
        self.best_x = None

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, y = [], []
        
        # Latin Hypercube Sampling (LHS) for better space coverage than random
        n_init = min(20, self.budget // 5)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x_scaled = sample[i]
                x = lb + x_scaled * (ub - lb)
            else:
                # Acquisition function: Expected Improvement
                y_min = np.min(y)
                def acq(x_scaled):
                    x_raw = lb + x_scaled * (ub - lb)
                    mu, sigma = self.gp.predict(x_raw.reshape(1, -1), return_std=True)
                    sigma = np.maximum(sigma, 1e-12)
                    imp = y_min - mu
                    z = imp / sigma
                    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))

                # Multi-start optimization of the acquisition function
                best_acq = np.inf
                x_best_acq = np.random.rand(self.dim)
                for _ in range(8):
                    x0 = np.random.rand(self.dim)
                    res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq, x_best_acq = res.fun, res.x
                x = lb + x_best_acq * (ub - lb)

            try:
                f = float(func(x))
            except Exception:
                # Fallback to random point if evaluation fails
                x = lb + np.random.rand(self.dim) * (ub - lb)
                f = float(func(x))
                
            if f < self.min_y:
                self.min_y, self.best_x = f, x
            
            X.append(x)
            y.append(f)
            
            # Refit GP periodically. Frequency increases as more data is gathered.
            if len(X) >= n_init and (len(X) < 30 or len(X) % 5 == 0):
                try:
                    self.gp.fit(np.array(X), np.array(y))
                except:
                    pass # Keep old hyperparameters if fit fails
        
        return self.min_y, self.best_x