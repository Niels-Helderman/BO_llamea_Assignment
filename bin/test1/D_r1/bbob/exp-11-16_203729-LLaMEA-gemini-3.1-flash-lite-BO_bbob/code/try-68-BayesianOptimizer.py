import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-8)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, normalize_y=True, n_restarts_optimizer=0)
        self.length = 0.5

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for robust space-filling initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        n_init = max(2 * self.dim + 2, 12)
        X = sampler.random(n_init)
        y = np.array([func(x * (ub - lb) + lb) for x in X])
        
        best_idx = np.argmin(y)
        best_y, best_x_norm = float(y[best_idx]), X[best_idx].copy()
        
        for i in range(n_init, self.budget):
            try:
                # Refit hyperparameters every few steps to balance cost vs accuracy
                self.gp.n_restarts_optimizer = 1 if i % 8 == 0 else 0
                self.gp.fit(X, y)
            except:
                pass
            
            # Log-Expected Improvement for better numerical stability during optimization
            def log_ei(x):
                mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                diff = best_y - mu
                z = diff / sigma
                # Log-EI = log(diff * Phi(z) + sigma * phi(z))
                # Using logsumexp trick: log(A + B) = logA + softplus(logB - logA)
                log_a = np.log(np.maximum(diff, 1e-20)) + norm.logcdf(z)
                log_b = np.log(sigma) + norm.logpdf(z)
                return -np.log(np.exp(log_a) + np.exp(log_b))
            
            # Trust region search to balance exploitation and exploration
            lb_tr = np.clip(best_x_norm - self.length, 0, 1)
            ub_tr = np.clip(best_x_norm + self.length, 0, 1)
            
            next_x = None
            best_acq = np.inf
            # Multi-start acquisition optimization
            for _ in range(4):
                x0 = np.random.uniform(lb_tr, ub_tr)
                res = minimize(log_ei, x0, bounds=list(zip(lb_tr, ub_tr)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq, next_x = res.fun, res.x
            
            # Fallback to random sampling if surrogate optimization struggles
            if next_x is None:
                next_x = np.random.uniform(0, 1, self.dim)
            
            val = func(next_x * (ub - lb) + lb)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            # Update trust region size based on success
            if val < best_y:
                best_y, best_x_norm = float(val), next_x
                self.length = np.clip(self.length * 1.1, 0.02, 1.0)
            else:
                self.length = np.clip(self.length * 0.9, 0.02, 1.0)
                
        return best_y, best_x_norm * (ub - lb) + lb