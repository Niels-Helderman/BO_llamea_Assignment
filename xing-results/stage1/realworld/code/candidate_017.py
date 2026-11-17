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
        # Use Matern 5/2 as it is well-suited for general black-box objective landscapes.
        # Add WhiteKernel to explicitly model observation noise.
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=3, normalize_y=True)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling than purely random sampling
        n_init = max(2 * self.dim + 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = qmc.scale(sampler.random(n_init), lb, ub)
        
        for i in range(self.budget):
            if i < n_init:
                x_next = init_points[i]
            else:
                x_next = self._propose_next(lb, ub, i)
            
            x_next = np.clip(x_next, lb, ub)
            y_val = float(func(x_next))
            
            self.X.append(x_next)
            self.y.append(y_val)
            
            # Re-fit the GP after each point, but n_restarts is kept low for speed
            self.gp.fit(np.array(self.X), np.array(self.y))

        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx]

    def _propose_next(self, lb, ub, iteration):
        # Use Upper Confidence Bound (UCB) for adaptive exploration vs exploitation
        # beta decays linearly to converge on the found optima
        beta = 2.0 * np.sqrt(np.log(iteration + 1))
        
        def acq(x):
            mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return (mu - beta * std)[0]

        best_x = np.random.uniform(lb, ub, self.dim)
        min_acq = np.inf
        
        # Optimize acquisition function with multiple random starts
        for x0 in [np.random.uniform(lb, ub, self.dim) for _ in range(5)]:
            res = minimize(acq, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < min_acq:
                min_acq = res.fun
                best_x = res.x
        
        # Fallback if optimization fails
        if not np.isfinite(best_x).all():
            return np.random.uniform(lb, ub, self.dim)
        return best_x