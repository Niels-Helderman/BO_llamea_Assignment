import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.y = []
        # Use Matern 5/2 with ARD and integrated noise handling
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(dim)) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=0, normalize_y=False)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        n_init = min(15, self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose(lb, ub, i)
            
            x_next = np.clip(x_next, lb, ub)
            val = float(func(x_next))
            self.X.append(x_next)
            self.y.append(val)
            
            if (i + 1) % 5 == 0 or i == n_init - 1:
                X_arr = np.array(self.X)
                y_arr = np.array(self.y)
                self.y_m, self.y_s = np.mean(y_arr), np.std(y_arr) + 1e-8
                self.gp.fit(X_arr, (y_arr - self.y_m) / self.y_s)
                # Re-optimize kernel hyperparameters occasionally
                if i % 50 == 0: self.gp.n_restarts_optimizer = 3
                else: self.gp.n_restarts_optimizer = 0

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose(self, lb, ub, iteration):
        # UCB acquisition: kappa decays to focus exploitation over time
        kappa = 2.0 * np.sqrt(np.log(iteration + 2))
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return (mu - kappa * std)[0]

        x_best = np.random.uniform(lb, ub, self.dim)
        min_acq = np.inf
        
        # Multi-start optimization
        for x0 in [np.random.uniform(lb, ub, self.dim) for _ in range(5)]:
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < min_acq:
                min_acq, x_best = res.fun, res.x
        return x_best