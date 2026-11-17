import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use Matern 5/2 as it is well-suited for BBOB functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=0.0, 
            normalize_y=True, 
            n_restarts_optimizer=2, 
            optimizer='fmin_l_bfgs_b'
        )

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Latin Hypercube Sampling (LHS) approximation
        def get_lhs(n, dim):
            samples = np.zeros((n, dim))
            for i in range(dim):
                samples[:, i] = np.random.permutation(np.linspace(0, 1, n))
            return samples * (ub - lb) + lb

        # Use 2*dim + 2 for initial design to balance exploration/surrogate fidelity
        init_size = min(max(2 * self.dim + 2, 8), self.budget)
        X = get_lhs(init_size, self.dim)
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]

        def get_ei(x, gp, y_min):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            z = (y_min - mu) / sigma
            return -((y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))

        for i in range(init_size, self.budget):
            try:
                # Refit hyperparameters every iteration; n_restarts is low to save budget
                self.gp.fit(X, y)
                
                y_min = np.min(y)
                best_acq = np.inf
                next_x = None
                
                # Multi-start acquisition optimization
                for _ in range(8):
                    x0 = np.random.uniform(lb, ub)
                    res = minimize(get_ei, x0, args=(self.gp, y_min), 
                                   bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.fun < best_acq:
                        best_acq = res.fun
                        next_x = res.x
                
                if next_x is None:
                    next_x = np.random.uniform(lb, ub)
            except:
                # Fallback for numerical instability
                next_x = np.random.uniform(lb, ub)
                
            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
                
        return float(best_y), best_x