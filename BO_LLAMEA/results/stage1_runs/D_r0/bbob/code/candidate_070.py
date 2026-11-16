import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class BayesianOptimizer:
    def __init__(self, budget=600, dim=6):
        self.budget = budget
        self.dim = dim
        # ARD Matern kernel: length_scale is a vector of dim length
        self.kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=2)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Sobol sequence for robust space filling
        n_init = 2 * self.dim + 4
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_obs = qmc.scale(sampler.random(n_init), lb, ub).tolist()
        y_obs = [func(x) for x in X_obs]
        
        best_x = X_obs[np.argmin(y_obs)]
        best_y = float(np.min(y_obs))
        
        for i in range(n_init, self.budget):
            X_data = np.array(X_obs)
            y_data = np.array(y_obs)
            
            # Fit GP with ARD (Automatic Relevance Determination)
            if i % 10 == 0:
                self.gp.n_restarts_optimizer = 3
            else:
                self.gp.n_restarts_optimizer = 0
            
            try:
                self.gp.fit(X_data, y_data)
            except:
                pass
            
            # Expected Improvement Acquisition
            def ei(x):
                x = x.reshape(1, -1)
                mu, std = self.gp.predict(x, return_std=True)
                std = np.maximum(std, 1e-12)
                diff = best_y - mu
                z = diff / std
                return -(diff * norm.cdf(z) + std * norm.pdf(z))

            # Multi-start acquisition optimization
            next_x = None
            best_acq = np.inf
            
            # Candidates: Local search around best_x + Random search
            starts = [best_x] + [np.random.uniform(lb, ub) for _ in range(10)]
            for x0 in starts:
                res = minimize(ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    next_x = res.x
            
            if next_x is None:
                next_x = np.random.uniform(lb, ub)
            
            y_new = func(next_x)
            X_obs.append(next_x)
            y_obs.append(y_new)
            
            if y_new < best_y:
                best_y, best_x = float(y_new), next_x
                
        return best_y, np.array(best_x)