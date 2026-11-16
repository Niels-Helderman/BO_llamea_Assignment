import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5 

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for better space-filling initial coverage
        n_init = max(self.dim * 2, 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        # Beta schedule for UCB: starts high for exploration, decays for exploitation
        def get_beta(iter):
            return 2.0 * np.log(iter + 1)
        
        def get_ucb(x, gp, beta):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            return mu - np.sqrt(beta) * sigma

        for i in range(n_init, self.budget):
            try:
                self.gp.fit(X, y)
            except:
                pass
            
            beta = get_beta(i)
            lb_tr = np.clip(best_x_norm - self.length / 2, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length / 2, 0, 1)
            
            best_acq = np.inf
            next_x = None
            
            # Multi-start local optimization of the acquisition function
            for _ in range(8):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(lambda x: get_ucb(x, self.gp, beta), x0, 
                               bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B', tol=1e-3)
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None: 
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y = float(val)
                best_x_norm = next_x
                self.length = min(1.0, self.length * 1.1)
            else:
                self.length = max(0.02, self.length * 0.8)
            
            # Reset trust region if it contracts too much
            if self.length < 0.05 and (i % 10 == 0):
                self.length = 0.5
                
        return best_y, best_x_norm * (ub - lb) + lb