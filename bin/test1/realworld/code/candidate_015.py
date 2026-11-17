import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        X, Y = [], []
        best_y, best_x = float('inf'), None
        
        n_init = min(max(2 * self.dim + 1, 5), self.budget // 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_samples = sampler.random(n_init)
        
        for i in range(self.budget):
            if i < n_init:
                x = lb + (ub - lb) * init_samples[i]
            else:
                X_norm = (np.array(X) - lb) / (ub - lb)
                y_min, y_std = np.min(Y), np.std(Y)
                y_std = y_std if y_std > 1e-9 else 1.0
                Y_norm = (np.array(Y) - np.mean(Y)) / y_std
                
                try:
                    self.gp.fit(X_norm, Y_norm)
                    def acq(x_n):
                        mu_n, std_n = self.gp.predict(x_n.reshape(1, -1), return_std=True)
                        mu = mu_n * y_std + np.mean(Y)
                        sigma = np.maximum(std_n * y_std, 1e-12)
                        imp = y_min - mu
                        Z = imp / sigma
                        return -(imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                    best_ei, x_next = -1.0, np.random.rand(self.dim)
                    # Multi-start optimization
                    for _ in range(5):
                        x0 = np.random.rand(self.dim)
                        res = minimize(acq, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
                        if -res.fun > best_ei:
                            best_ei, x_next = -res.fun, res.x
                    x = lb + (ub - lb) * x_next
                except:
                    x = lb + (ub - lb) * np.random.rand(self.dim)
            
            x = np.clip(x, lb, ub)
            y = float(func(x))
            if y < best_y:
                best_y, best_x = y, x
            X.append(x)
            Y.append(y)
            
        return float(best_y), best_x