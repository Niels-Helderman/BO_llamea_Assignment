import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X_hist = []
        self.y_hist = []
        # Robust kernel with length-scale learning and noise adaptation
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=0)
        self.n_init = max(2 * self.dim + 2, 12)

    def _ucb(self, x_norm, beta):
        x = x_norm.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        return -(mu - beta * sigma)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        best_y = np.inf
        best_x = None

        for i in range(self.budget):
            if i < self.n_init:
                # Latin Hypercube-like design using jittered grid
                x = lb + (ub - lb) * (np.random.rand(self.dim) * 0.9 + (i % self.n_init) / self.n_init * 0.1)
                x = np.clip(x, lb, ub)
            else:
                # Update GP model
                X_train = np.array(self.X_hist)
                y_train = np.array(self.y_hist)
                try:
                    self.gp.fit(X_train, y_train)
                except:
                    pass
                
                # Adaptive UCB: Beta schedule promotes global exploration early, exploitation late
                beta = 2.0 * np.sqrt(np.log(i + 1))
                
                best_res_val = np.inf
                x_next = np.random.rand(self.dim)
                
                # Global optimization of the acquisition function
                for _ in range(5):
                    x0 = np.random.rand(self.dim)
                    res = minimize(self._ucb, x0, args=(beta,), bounds=[(0, 1)]*self.dim, method='L-BFGS-B')
                    if res.fun < best_res_val:
                        best_res_val = res.fun
                        x_next = res.x
                
                x = x_next * (ub - lb) + lb
                x = np.clip(x, lb, ub)

            y = func(x)
            self.X_hist.append(x)
            self.y_hist.append(y)
            
            if y < best_y:
                best_y = y
                best_x = x
        
        return float(best_y), np.array(best_x)