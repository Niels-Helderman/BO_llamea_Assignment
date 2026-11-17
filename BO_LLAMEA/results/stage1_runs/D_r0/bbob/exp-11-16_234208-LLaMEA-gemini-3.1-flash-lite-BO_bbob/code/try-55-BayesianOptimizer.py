import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Robust kernel for deterministic functions
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=0)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial design: Sobol sequence for better space coverage
        n_init = min(self.budget, 2 * self.dim + 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        init_points = sampler.random(n_init)
        
        X_obs, y_obs = [], []
        
        for i in range(self.budget):
            if i < n_init:
                x = init_points[i] * (ub - lb) + lb
            else:
                # Update GP periodically; refit hyperparameters less frequently to save cost
                self.gp.fit(np.array(X_obs), np.array(y_obs))
                x = self._propose(lb, ub, i, self.budget)
            
            y = float(func(x))
            X_obs.append(x)
            y_obs.append(y)
            
        best_idx = np.argmin(y_obs)
        return float(y_obs[best_idx]), np.array(X_obs[best_idx])

    def _propose(self, lb, ub, iteration, total_budget):
        # Adaptive beta: starts high for exploration, decays for exploitation
        beta = 2.0 * np.sqrt(2 * np.log(iteration + 1))
        
        def ucb(x_norm):
            x = x_norm * (ub - lb) + lb
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            return -(mu - beta * sigma)

        # Multi-start optimization of the acquisition function
        best_val = np.inf
        best_x = np.random.uniform(0, 1, self.dim)
        
        # Candidate generation
        candidates = [np.random.uniform(0, 1, self.dim) for _ in range(20)]
        # Add current best location to encourage local refinement
        if len(self.gp.X_train_) > 0:
            best_observed = self.gp.X_train_[np.argmin(self.gp.y_train_)]
            candidates.append(best_observed)

        for x0 in candidates:
            try:
                res = minimize(ucb, x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B', tol=1e-3)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
                
        return best_x * (ub - lb) + lb