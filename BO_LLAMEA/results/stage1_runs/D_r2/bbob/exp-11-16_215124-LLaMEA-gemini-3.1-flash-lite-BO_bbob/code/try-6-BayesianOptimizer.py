import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # Robust kernel: Matern 5/2 is standard for smooth functions
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)
        
    def _cma_es_ei(self, best_y, lb, ub, range_val):
        """Simple CMA-ES inspired acquisition optimizer to find max of EI."""
        pop_size = 4 + int(3 * np.log(self.dim))
        x_mean = np.random.rand(self.dim)
        sigma = 0.3
        best_ei_val = -np.inf
        best_ei_x = x_mean
        
        for _ in range(20):
            pop = np.clip(x_mean + sigma * np.random.randn(pop_size, self.dim), 0, 1)
            mu, std = self.gp.predict(pop, return_std=True)
            std = np.maximum(std, 1e-12)
            imp = best_y - mu
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            
            idx = np.argmax(ei)
            if ei[idx] > best_ei_val:
                best_ei_val = ei[idx]
                best_ei_x = pop[idx]
                x_mean = pop[idx]
            else:
                sigma *= 0.95
                
        return lb + best_ei_x * range_val

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_val = ub - lb
        
        init_size = min(20, self.budget // 5)
        X = np.random.uniform(0, 1, (init_size, self.dim)) * range_val + lb
        y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(y)
        best_x, best_y = X[best_idx].copy(), y[best_idx]
        
        for i in range(init_size, self.budget):
            try:
                # Periodic refit to balance overhead and accuracy
                if (i - init_size) % 5 == 0:
                    self.gp.fit(X, (y - np.mean(y)) / (np.std(y) + 1e-9))
                
                next_x = self._cma_es_ei(best_y, lb, ub, range_val)
            except:
                next_x = np.random.uniform(lb, ub)
            
            # Ensure no duplicates
            if np.any(np.all(np.isclose(X, next_x), axis=1)):
                next_x = np.random.uniform(lb, ub)

            val = func(next_x)
            X = np.vstack([X, next_x])
            y = np.append(y, val)
            
            if val < best_y:
                best_y, best_x = val, next_x
            
        return float(best_y), best_x