# PURPOSE AND REMAINING WORK
#
# This module should contain the callable optimization methods used throughout
# the assignment. All methods should eventually follow one documented calling
# convention so that generated algorithms, CMA-ES, BoTorch, and the optional
# random-search smoke test can be benchmarked under identical conditions.
#
# The copied imports from the acquisition-functions notebook provide the
# BoTorch, GPyTorch, IOH, Sobol, and acquisition-optimization dependencies used
# by the copied BayesianOptimizer class. The copied AbstractOptimizer class
# provides the course's common optimizer shape. The copied random_search_alg
# block is a simple reference implementation. The copied acquisition-function
# and BayesianOptimizer blocks provide course material for surrogate- and
# acquisition-based BO and can be studied or reused when constructing the
# BoTorch-style method. They are not yet a complete, assignment-configured
# baseline interface.
#
# Still missing:
# - a shared method invocation and result format;
# - the instructor-specified BoTorch baseline;
# - the CMA-ES baseline;
# - loading and invoking the two selected generated algorithms;
# - consistent seed, initialization, minimization, and evaluation-budget
#   handling for every method;
# - recording the complete best-so-far trajectory rather than only a final
#   best value where the copied method does not already provide it.
# - adapting the copied BayesianOptimizer to the instructor-specified
#   BoTorch baseline and assignment evaluation interface.

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 1.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday October 3, 2025

@authors: Elena Raponi, Lennard Froma, Ivan Olarte-Rodriguez

Revised on Thursday October 8, 2026

@authors: Elena Raponi, Ivan Olarte-Rodriguez, Santiago García-Mayayo

"""
# Import the Libraries

# Typical Libraries for Numerical Processing
import numpy as np

from typing import Callable

# Import the PyTorch Libraries
import torch
from torch import Tensor
from torch.distributions import Normal

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.models.transforms import Normalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel
from gpytorch.means import ConstantMean, ZeroMean, Mean
from gpytorch.constraints import Interval, GreaterThan, Positive


# To handle partial
from functools import partial

# For sampling distributions with scipy
from scipy.stats import norm

# Import the os library to handle file paths and the math library for mathematical operations
import os.path
import math

from matplotlib import pyplot as plt

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 22.
# Imports IOH get problem 
from ioh import get_problem
from ioh.iohcpp.problem import RealSingleObjective

# Imports the IOH Logging environments
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.trigger import Trigger, ALWAYS

# Typing Libraries
from typing import Iterable,Union,List,Tuple,Callable, Optional

# Import numpy, warnings and torch libraries 
import numpy as np
import warnings
import torch 
from torch import Tensor, device, float64 #This is relevant for the device to be used (CPU or GPU) and the type of tensor

# Import the BoTorch Libraries
from torch.quasirandom import SobolEngine # To use in this case for the DoE step
from botorch.models.transforms.input import Normalize # This is to normalize the inputs (from 0 to 1 in each dimension)
from botorch.models.transforms.outcome import Standardize # This is to standardize the outputs (0 mean and unit variance)
from botorch.optim import optimize_acqf # This is to optimize the acquisition function
from botorch.acquisition import AcquisitionFunction # This is the base class for the acquisition functions
from botorch.acquisition.analytic import AnalyticAcquisitionFunction # This is the base class for the analytic acquisition functions

# This is to import the Expected Improvement and other acquisition functions
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, LogProbabilityOfImprovement

# this is to import the automatic fitting method to use and 'train' the GP hyperparameters via Maximum Likelihood Estimation
from botorch.fit import fit_gpytorch_mll

# This is to call the decorator to handle the input transformations for the case of batch evaluations in Acquisition Functions
from botorch.utils.transforms import t_batch_mode_transform

from botorch.exceptions import BadInitialCandidatesWarning, ModelFittingError

# This is to import the GPyTorch libraries
from torch.distributions.normal import Normal # This is to handle the normal distribution (for the acquisition functions)

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 10.
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

class AbstractOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    def __init__(self, 
                 dim:int, 
                 lb:Optional[Union[int,float,np.ndarray]]=-5.0, 
                 ub:Optional[Union[int,float,np.ndarray]]=5.0,
                 **kwargs)->None:
        '''
        Parameters
        ----------
        
        dim : int
            The dimensionality of the input space.
        lb : int, float or np.ndarray
            The lower bound(s) for the input space.
        ub : int, float or np.ndarray
            The upper bound(s) for the input space.
        **kwargs : dict
            Additional hyperparameters for the optimizer.

        '''

        self.dim = dim
        self.hyperparameters = kwargs
        self.lb = lb
        self.ub = ub
        ...

    @abstractmethod
    def __call__(self, 
                 func:Callable, 
                 budget:int, 
                 random_seed:int=43, 
                 **kwds)->dict:
        '''
        Parameters
        ----------
        func : callable
            The objective function to be minimized.
        budget : int
            The total number of function evaluations allowed.
        random_seed : int
            Seed for random number generator to ensure reproducibility.
        **kwds : dict
            Additional parameters for the optimizer. Normally assigned to hardware settings.
        Returns
        -------
        dict
            A dictionary containing the optimization results.
        '''

        assert callable(func), "The objective function must be callable."
        assert isinstance(budget, int) and budget > 0, "Budget must be a positive integer."
        assert isinstance(random_seed, int), "Random seed must be an integer."
        

        ...
    
    @abstractmethod
    def run(self, func:Callable, budget:int, random_seed=43, **kwds)->dict:
        '''
        A wrapper method to call the optimizer.
        '''
        self.__call__(func, budget, random_seed, **kwds)
    

    @property
    def dim(self)->int:
        """
        The dimensionality of the input space.
        """
        return self._dim
    
    @dim.setter
    def dim(self, d:int)->None:
        if not isinstance(d, int) or d <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self._dim = d
    
    @property
    def lb(self)->np.ndarray:
        '''
        The lower bound(s) for the input space.
        '''
        return self._lb
    
    @lb.setter
    def lb(self, lb:Union[int,float,np.ndarray])->None:
        if isinstance(lb, (int, float)):
            self._lb = np.full(self.dim, lb)
        elif isinstance(lb, np.ndarray):
            if lb.shape != (self.dim,):
                raise ValueError(f"Lower bounds must be a numpy array of shape ({self.dim},).")
            self._lb = lb
        else:
            raise ValueError("Lower bounds must be a float, int, or numpy array.")
    
    @property
    def ub(self)->np.ndarray:
        return self._ub
    
    @ub.setter
    def ub(self, ub:Union[int,float,np.ndarray])->None:
        if isinstance(ub, (int, float)):
            self._ub = np.full(self.dim, ub)
        elif isinstance(ub, np.ndarray):
            if ub.shape != (self.dim,):
                raise ValueError(f"Upper bounds must be a numpy array of shape ({self.dim},).")
            self._ub = ub
        else:
            raise ValueError("Upper bounds must be a float, int, or numpy array.")
        
        if np.any(self._ub <= self.lb):
            raise ValueError("Each upper bound must be greater than the corresponding lower bound.")
    
    @property
    @abstractmethod
    def name(self)->str:
        '''
        The name of the optimization algorithm.
        '''
        ...


# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 12.
# This is a simple algorithm (Random Search) to have a standard call.
def random_search_alg(problem: Union[RealSingleObjective, Callable], budget:int=100, seed:int=42,
                         dimension:int=2, minimize:bool=True):
    """A simple random search algorithm for demonstration purposes.

    Args:
        problem: The optimization problem to solve.
        budget: The maximum number of evaluations allowed. Default is 100.
        seed: The random seed for reproducibility. Default is 42.
        dimension: The dimension of the search space. Default is 2.
        minimize: Whether to minimize or maximize the objective function. Default is True.

    Returns:
        best_x: The best solution found.
        best_y: The best objective value found.
        best_idx: The index of the best solution in the search history.
    """

    # Generate a random number generator with the given seed for reproducibility
    rng = np.random.default_rng(seed=seed)

    if isinstance(problem, RealSingleObjective):
        lbs = np.asarray(problem.bounds.lb)
        ubs = np.asarray(problem.bounds.ub)
        dimension_ = problem.meta_data.n_variables
    else:
        assert dimension > 0 and isinstance(dimension, int), "Dimension must be a positive integer."
        lbs = np.full(dimension, -5.0)
        ubs = np.full(dimension, 5.0)
        dimension_ = dimension

    # Initialize the placeholders for the best x and best y values
    best_x = None
    best_y = np.inf if minimize else -np.inf
    best_idx = -1

    for ii in range(budget):

        # Generate a random solution within the problem's bounds
        random_solution = rng.uniform(lbs, ubs, size=(dimension_,)).tolist()
        
        # Evaluate the random solution
        y = problem(random_solution)

        # Update the best solution if the current one is better
        if (minimize and y < best_y) or (not minimize and y > best_y):
            best_y = y
            best_idx = ii
            best_x = random_solution

    return best_x, best_y, best_idx

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 9.

def EI(m:float,v:float,best_y:float, minimization:bool = False)-> float:
    
    '''
    This is a base Expected Improvement function. The inputs of this function are:
    - m: The posterior mean
    - v: The posterior variance
    - best_y: The sampled best value of the objective
    - minimization: A boolean handle to determine if the objective is minimized or maximized
    '''
        
    ##### TODO: Enter your code here:

    if minimization:
        delta = best_y - m
    else:
        delta =  m - best_y
        
        
    s = np.sqrt(v).ravel()[0]
    z = delta/s if s >1e-08 else delta
    ret = max(0.0, delta)
    ret += s * norm.pdf(z, loc=0) 
    ret -= abs(delta)*norm.cdf(-1*abs(z))
    
    return ret
    
    ##### --------------- END OF CODE INPUT -----------------------



# Test your code!!!
print(EI(1,1e-100,0))
print(EI(2,1,0))
print(EI(0,2,2))
print(EI(0,10,2))

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 13.
def ucb(m:float,v:float,best_y:float, psi=1.96):
    """The upper confidence acquisition function."""
    if v <= 0:
        return max(0,m-best_y)

    return m + psi * np.sqrt(v)

def lcb(m:float,v:float,best_y:float, psi=1.96):
    """The lower confidence acquisition function."""
    if v <= 0:
        return max(0,m-best_y)
    
    return m - psi * np.sqrt(v)

def poi(m:float,v:float,best_y:float, psi=0.):
    """Return the probability of improvement.
    
    Arguments
    psi   -- A parameter that controls exploration.
    """
    if v <= 0:
        return max(0,m-best_y)
    
    return norm.cdf((m - best_y - psi) / np.sqrt(v))

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 24.
# This is a class defining a Bayesian Optimizer
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

class BayesianOptimizer:
    r"""
    This class defines an object to perform Optimization in a complete Bayesian Loop from the 
    exact definition.
    """

    def __init__(self, 
                 acquisition_function:AnalyticAcquisitionFunction = LogExpectedImprovement, # The log of the Expected Improvement 
                                                                                            # is used to avoid numerical issues
                 mean:Callable = ConstantMean(),
                 kernel:Callable = ScaleKernel(MaternKernel(nu=2.5, 
                                                            lengthscale_constraint=GreaterThan(1e-3), # This constraints the lengthscale
                                                            raw_lengthscale_constraint=Positive()
                                                            ),
                                                outputscale_constraint=Interval(1e-3, 1e3)
                                              )
                )->None:
        r"""
        This is an the initialization of the class. 

        Args
        ------------
        acquisition_function:AnalyticAcquisitionFunction
            This is the acquisition function to be used in the optimization process. It must be an instance of the
            AnalyticAcquisitionFunction class from BoTorch
        mean:Callable
            This is the mean function to be used in the GP model. It must be an instance of the gpytorch.means.Mean class
        kernel:Callable
            This is the kernel function to be used in the GP model. It must be an instance of the gpytorch.kernels.Kernel class
        ------------
        """
        # Store the acquisition function
        self.__original_acquisition_function = acquisition_function

        # Initialize the arrays to store the sampled values
        self.X:Tensor = []
        self.fX:Tensor = []

        # use a GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        

        # Store the Analytic acquisition function definition as an attribute of the class
        assert issubclass(acquisition_function, AnalyticAcquisitionFunction), "The acquisition function must be an instance"\
                                              "of the AnalyticAcquisitionFunction class"
        self.__original_acquisition_function:AnalyticAcquisitionFunction = acquisition_function

        # Store the kernel and mean functions
        assert issubclass(type(kernel),Kernel), "The kernel must be an instance of the gpytorch.kernels.Kernel class"
        self.__kernel:Callable = kernel 

        assert issubclass(type(mean),Mean), "The mean must be an instance of the gpytorch.means.Mean class"
        self.__mean:Callable = mean

    
    def run(self,problem_instance:RealSingleObjective, 
                 budget:int = 100, 
                 initial_doe_samples:int = 10,
                 random_seed:int = 43)->None:
        r'''
        This method will perform the optimization process. It will use the acquisition function, kernel and mean
        functions defined in the initialization of the class.
        '''

        self.__call__(problem_instance=problem_instance,
                        budget=budget,
                        initial_doe_samples=initial_doe_samples,
                        random_seed=random_seed)
        


    
    def __call__(self, 
                 problem_instance:RealSingleObjective, 
                 budget:int = 100, 
                 initial_doe_samples:int = 10,
                 random_seed:int = 43)->None:
        
        r'''
        This method will perform the optimization process. It will use the acquisition function, kernel and mean
        functions defined in the initialization of the class.

        Args
        ------------
        problem_instance:RealSingleObjective
            This is an instance of the ioh.iohcpp.problem.RealSingleObjective class. It contains the definition of the objective and possible constraint functions
            to be optimized
        budget:int
            This is the total budget of function evaluations to be used in the optimization process
        initial_doe_samples:int
            This is the number of initial samples to be used in the optimization process
        random_seed:int
            This is the random seed to be used in the optimization process
        
        '''

        #### TODO: Implement the optimization loop for the Bayesian Optimizer. The steps are:
        # 1. Get the initial DoE samples using a Sobol sequence
        # 2. Transform the initial DoE samples to the problem bounds
        # 3. Evaluate the objective function at the initial DoE samples
        # 4. Fit the initial model
        # 5. Perform a loop until the budget is exhausted:
        # 5.1. Extract new candidates using the acquisition function
        # 5.2. Evaluate the objective function at the new candidates
        # 5.3 Update the sampled values
        # 5.4. Refit the model

        # Attach the problem instance
        self.problem = problem_instance

        self.budget = budget

        initial_doe = self.initial_doe_samples(n_samples=initial_doe_samples,
                                               n_variables=self.problem.meta_data.n_variables,
                                               random_seed=random_seed)
        
        self.X = initial_doe.to(device=self.device, dtype=self.dtype)

        # Evaluate the model at the initial DOE
        fX = torch.tensor([self.problem(x_i.detach().cpu().numpy()) for x_i in self.X])

        self.fX = fX.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        
        # Fit initial model
        _, gp_model = self.fit_model(train_X=self.X,
                                train_obj=self.fX) 
        
        
        # Perform a loop until the possible runs are over!
        #Implement for students
        for _ in range(budget - initial_doe_samples):
            
            # Extract new candidates
            new_x_i = self.optimize_acqf_and_get_observation(gp_model=gp_model)
            
            # Compute again the function
            x_i, f_i_evals= self.evaluate_function(new_x_i)

            # Update the sampled values
            self.X = torch.cat((self.X,x_i),dim=0)
            self.fX = torch.cat((self.fX,f_i_evals),dim=0)

            # Refit the model again!
            _, gp_model = self.fit_model(train_X=self.X,
                                train_obj=self.fX)

        # END OF TODO
        
        # Return the best found solution
        best_idx = torch.argmin(self.fX)

        print("The best solution found is: ")
        print("Best idx = ", best_idx.cpu().numpy()+1)
        print("x = ", self.X[best_idx,:].cpu().numpy())
        print("f(x) = ", self.fX[best_idx].cpu().numpy())


    def initial_doe_samples(self, n_samples:int, n_variables:int, random_seed:int = 43)->Tensor:
        r'''
        This method will generate the initial DoE samples using a Sobol sequence. 
        Also the samples will be transformed to the problem bounds. The output will be a tensor of size (n_samples, n_variables) containing the initial DoE samples.


        Args
        ------------
        n_samples:int
            This is the number of initial samples to be generated
        n_variables:int
            This is the number of variables of the problem
        random_seed:int
            This is the random seed to be used in the optimization process

        Returns
        ------------
        initial_doe:Tensor
            This is a tensor of size (n_samples, n_variables) containing the initial DoE samples within the problem bounds.
        '''
        # Get the initial doe
        sampler = SobolEngine(dimension=n_variables,
                                seed=random_seed, scramble=True)
        
        initial_doe:Tensor = sampler.draw(n_samples, dtype=self.dtype).to(device=self.device)

        # Transform the initial doe to the problem bounds
        #Implementation for students
        for ii in range(n_variables):
            initial_doe[:,ii] = self.bounds[0,ii] + (self.bounds[1,ii] - self.bounds[0,ii]) * initial_doe[:,ii]

        initial_doe = initial_doe.reshape((n_samples,
                                           n_variables))

        return initial_doe
          
           
    def fit_model(self,
                  train_X:Tensor, 
                  train_obj:Tensor,
                  optimizer_steps:int=100)->Tuple[ExactMarginalLogLikelihood,SingleTaskGP]:
        
        """
        This overload will use the same definition as before. The output is a ExactMarginalLogLikelihood object and
        a SingleTaskGP Object. In the case of the SingleTaskGP object, it will correspond to the GP of the objective function
        
        Args
        ------------
        train_X:Tensor
            This is a tensor of size (n_samples, n_features) containing the sampled input locations
        train_obj:Tensor
            This is a tensor of size (n_samples, 1) containing the sampled objective function
        optimizer_steps:int
            This is the number of steps to be used in the optimization of the hyperparameters of the GP model. The default value is 100
        ------------

        Returns
        ------------
        mll:ExactMarginalLogLikelihood
            This is the Exact Marginal Log Likelihood object corresponding to the objective function GP model
        model_obj:SingleTaskGP
            This is the Single Task GP model corresponding to the objective function
        """


        # Define a fixed noise likelihood with very small noise to emulate a noisefree model
        noisefree_variance = 0.0001
        noises = torch.ones(len(train_obj)) * noisefree_variance
        noise_free_likelihood = FixedNoiseGaussianLikelihood(noise=noises)

        
        # Define models for objective and constraint
        model_obj = SingleTaskGP(
            train_X=train_X,
            train_Y=train_obj,
            ### NOTE: These modules are used to normalize the inputs and standardize the outputs for better numerical stability
            input_transform=Normalize(d=train_X.shape[-1],
                                      bounds=self.bounds),
            outcome_transform=Standardize(m=train_obj.shape[-1]),
            ### END OF NOTE
            likelihood= noise_free_likelihood,
            mean_module=self.mean,
            covar_module=self.kernel

        ).to(device=self.device)

        # Wrap the model into a Marginal Log Likelihood Computation
        mll = ExactMarginalLogLikelihood(likelihood=model_obj.likelihood,
                                         model=model_obj)

        # Train the model(s) like in the tutorial
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            # USe a fallback and use manual optimization of the hyperparameters
            warnings.warn("Model fitting failed. Using fallback optimization.", ModelFittingError)
            # Set the model to training mode
            model_obj.train()
            # Use the Adam optimizer to optimize the hyperparameters
            optimizer = torch.optim.Adam(model_obj.parameters(), lr=0.1)

            # Use the marginal log likelihood as the loss function
            mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
            # Perform a few steps of optimization
            for _ in range(optimizer_steps):
                optimizer.zero_grad()
                output = model_obj(train_X)
                loss = -mll(output, train_obj)
                loss.backward()
                optimizer.step()

        finally:
            model_obj.eval()

        return mll, model_obj
    
    def reset(self)->None:
        r'''
        This method will reset the sampled values and the problem instance
        '''
        self.X = []
        self.fX = []
        self.__problem = None
        self.__bounds = None

      
    
    def optimize_acqf_and_get_observation(self,gp_model:SingleTaskGP)->Tensor:
        
        """
        This function is responsible of using the fitted GP Model and get the new possible values .

        Args
        ------------
        gp_model:SingleTaskGP
            This is the fitted Single Task GP model corresponding to the objective function
        ------------
        """
        # Get the best function evaluation
        best_f = self.fX.min().item()
        objective_acq = self.acq_fun(model=gp_model, 
                                    best_f=best_f,
                                    maximize=False)


        
        # NOTE: This is set since we are only looking for one candidate at each iteration since 
        #       we're using Analytic Acquisition Functions. For batch acquisition functions, this should be
        #       changed to the corresponding batch size set by the user when initializing the Acquisition Function

        q_samples:int = 1


        # Pass the Constrained Expected Improvement function to the Default BoTorch Acquisition Function Optimizer
        candidate, acq_value = optimize_acqf(
                                                acq_function=objective_acq,
                                                bounds=self.bounds,
                                                q=q_samples,
                                                num_restarts=20,
                                                raw_samples=512,
                                        )
        
        return candidate.detach().reshape(shape=(q_samples,self.problem.meta_data.n_variables))
    
    def evaluate_function(self,new_x:Tensor)->Tuple[Tensor,Tensor]:
        r"""
        This function evaluates the objective and constraint functions at the new possible candidates
        """
        # Evaluate the objective function
        f_evals = torch.tensor([self.problem(x_i.detach().cpu().numpy()) for x_i in new_x],
                               device=self.device,
                               dtype=self.dtype).unsqueeze(-1)
        
        return new_x, f_evals   

    @property
    def acq_fun(self)->AnalyticAcquisitionFunction:
        r'''
        This is a property to get the acquisition function
        '''
        return self.__original_acquisition_function
    
    @acq_fun.setter
    def acq_fun(self, acquisition_function:AnalyticAcquisitionFunction)->None:
        assert issubclass(acquisition_function, AnalyticAcquisitionFunction, "The acquisition function must be"\
                                                                              " an instance of the AnalyticAcquisitionFunction class"
                        )   
        self.__original_acquisition_function = acquisition_function
    
    @property
    def kernel(self)->Callable:
        return self.__kernel
    
    @kernel.setter
    def kernel(self, kernel:Callable)->None:
        assert issubclass(type(kernel),Kernel), "The kernel must be an instance of the gpytorch.kernels.Kernel class"
        self.__kernel = kernel
    
    @property
    def mean(self)->Callable:
        return self.__mean
    
    @mean.setter
    def mean(self, mean:Callable)->None:
        assert issubclass(type(mean),Mean), "The mean must be an instance of the gpytorch.means.Mean class"
        self.__mean = mean
    
    @property
    def problem(self)->RealSingleObjective:
        r'''
        This is a property to get the problem instance
        '''
        return self.__problem
    
    @problem.setter
    def problem(self, problem_instance:RealSingleObjective)->None:
        '''

        Args
        ------------
        problem_instance:Problem
            This is an instance of the cocoex.Problem class. It contains the definition of the
            objective and constraint functions to be optimized
        ------------
        '''
        assert isinstance(problem_instance, RealSingleObjective), "The problem instance must be an instance of the"\
                                                                  "ioh.iohcpp.problem.RealSingleObjective class"
        self.__problem = problem_instance

        # Store the bounds of the problem
        self.__bounds = torch.tensor([problem_instance.bounds.lb.ravel(),
                                    problem_instance.bounds.ub.ravel()],
                                   device=self.device,
                                   dtype=self.dtype)

    @property
    def best_index(self)->int:
        r'''
        This is a property to get the index of the best solution found
        '''
        return torch.argmin(self.fX).item()
    
    @property
    def bounds(self)->Tensor:
        return self.__bounds

    @property
    def current_index(self)->int:
        r'''
        This is a property to get the current index of the best solution found
        '''
        return self.fX.shape[0]

    @property
    def budget(self)->int:
        r'''
        This is a property to get the budget of the optimization process
        '''
        return self.__budget

    @budget.setter
    def budget(self, budget:int)->None:
        assert isinstance(budget, int), "The budget must be an integer"
        self.__budget = budget

# Copied verbatim from Relevent Files/Acquisition_Functions_Solutions.ipynb, code cell 28.
class BayesianOptimizerWScheduler(BayesianOptimizer):
    r"""
    This class defines an object to perform Optimization in a complete Bayesian Loop from the 
    exact definition. It extends the BayesianOptimizer class to include a scheduler for the acquisition function.
    """

    def __init__(self, 
                 mean:Callable = ConstantMean(),
                 kernel:Callable = ScaleKernel(MaternKernel(nu=2.5, 
                                                            lengthscale_constraint=GreaterThan(1e-3), # This constraints the lengthscale
                                                            raw_lengthscale_constraint=Positive()
                                                            ),
                                                outputscale_constraint=Interval(1e-3, 1e3)
                                              ),
                 scheduler:Optional[Callable] = None
                )->None:
        r"""
        This is an the initialization of the class. 

        Args
        -----------
        mean:Callable
            This is the mean function to be used in the GP model. It must be an instance of the gpytorch.means.Mean class
        kernel:Callable
            This is the kernel function to be used in the GP model. It must be an instance of the gpytorch.kernels.Kernel class
        scheduler:Optional[Callable]
            This is an optional scheduler for the acquisition function. It should be a callable that takes in the current iteration and returns a modified acquisition function.
        ------------
        """
        acquisition_func = UpperConfidenceBound
        super().__init__(acquisition_function=acquisition_func, mean=mean, kernel=kernel)

        if scheduler is not None:
            assert callable(scheduler), "The scheduler must be a callable"
            self.scheduler = scheduler
        else:
            # Assign a cosine function scheduler as default
            self.scheduler = lambda iter_budget_pair : 3.98 * np.cos(np.pi/2 * iter_budget_pair[0] / iter_budget_pair[1])  + 0.01

    def optimize_acqf_and_get_observation(self, gp_model:SingleTaskGP)->Tensor:
        r"""
        This function is responsible for using the fitted GP Model and getting the new possible values, with an optional scheduler for the acquisition function.
        Given that we are using the Upper Confidence Bound (UCB) acquisition function, we can modify the exploration parameter based on the iteration number.

        Args
        ------------
        gp_model:SingleTaskGP
            This is the fitted Single Task GP model corresponding to the objective function.
        """

        
        # Get the best function evaluation
        best_f = self.fX.min().item()

        # Get the beta parameter from the scheduler based on the current iteration and budget
        beta = self.scheduler((self.current_index, self.budget))



        objective_acq = self.acq_fun(model=gp_model, 
                                    beta=beta,
                                    maximize=False)


        
        # NOTE: This is set since we are only looking for one candidate at each iteration since 
        #       we're using Analytic Acquisition Functions. For batch acquisition functions, this should be
        #       changed to the corresponding batch size set by the user when initializing the Acquisition Function

        q_samples:int = 1


        # Pass the Constrained Expected Improvement function to the Default BoTorch Acquisition Function Optimizer
        candidate, acq_value = optimize_acqf(
                                                acq_function=objective_acq,
                                                bounds=self.bounds,
                                                q=q_samples,
                                                num_restarts=20,
                                                raw_samples=512,
                                        )
        
        return candidate.detach().reshape(shape=(q_samples,self.problem.meta_data.n_variables))

# TODO: The source files do not provide complete CMA-ES and instructor-specified BoTorch baseline implementations with one shared interface.
