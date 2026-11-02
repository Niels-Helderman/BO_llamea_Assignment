# PURPOSE AND REMAINING WORK
#
# This module should own the assignment-specific configuration around the
# installed LLaMEA Stage 1 process. It should
# configure the selected free LLM provider, load the prompts, evaluate each
# generated candidate on the target problem for all ten seeds, return AOCC and
# execution feedback to LLaMEA, retain the best candidate, and persist the
# generated source, prompts, configuration, logs, and search history.
#
# It must use LLaMEA's existing evolutionary loop, population/solution
# objects, LLM interfaces, and experiment logging rather than reimplementing
# those framework components here.
#
# The copied Gemini_LLM and Groq_LLM blocks show two provider setup patterns.
# prepare_namespace, Solution, and ExperimentLogger are the building blocks
# for safely loading generated code and recording its evaluation. The copied
# evaluate_simple function is the main starting point for the candidate
# evaluator. The copied LLaMEA setup and run blocks show how to connect that
# evaluator to the outer evolutionary loop.
# The copied general-import block supplies the IOH, typing, NumPy, and
# environment dependencies used by those existing LLaMEA examples.
#
# Still missing:
# - configuration-file loading;
# - explicit BBOB and real-world target selection;
# - assignment budgets, ten assigned seeds, and dimension six;
# - robust per-candidate/per-seed result persistence;
# - selection and saving of the best solution for each target;
# - prompt loading from config/assignment.yaml;
# - explicit API-key validation and reproducible LLM settings;
# - a reusable function or command entry point for both searches.

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 4.
# Import the necessary libraries
import llamea
import numpy as np
from ioh import LogInfo
from ioh.iohcpp.logger import AbstractLogger
from ioh import get_problem
from ioh.iohcpp.problem import RealSingleObjective
from ioh.iohcpp.logger.trigger import ALWAYS
from copy import deepcopy, copy

from typing import Union, Tuple, List, Dict, Any, Optional, Callable
import os, sys


# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 19.
#gemini_api_key = os.environ.get("GEMINI_API_KEY")
from llamea import Gemini_LLM

gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")  # Replace with your actual Gemini API key

gemini_llm_obj = Gemini_LLM(gemini_api_key, "gemini-3.5-flash-lite") # Note that this depends on the Google

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 21.
from llamea import OpenAI_LLM

### WRAPPER CLASS
class Groq_LLM(OpenAI_LLM):
    """LLaMEA wrapper for Groq's OpenAI-compatible API."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.8,
        **kwargs,
    ):
        super().__init__(api_key, model, temperature, **kwargs)
        self.base_url = "https://api.groq.com/openai/v1"
        self._client_kwargs["base_url"] = self.base_url

groq_api_key = os.environ.get("GROQ_API_KEY","YOUR_GROQ_API_KEY_HERE")  # Replace with your actual Groq API key
groq_llm_obj = Groq_LLM(groq_api_key, "openai/gpt-oss-120b", temperature=0.8)

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 24.
# This imports the prepare_namespace function, which is used to create a safe execution environment for the generated code.
from llamea.utils import prepare_namespace

# This imports the Solution and ExperimentLogger classes, which are used to represent 
# the generated solution and log the evaluation process.
from llamea import Solution, ExperimentLogger

# Given that in our example to generate code, we will use torch, gpytorch, botorch, scipy, numpy, and math, we will allow these libraries in the namespace.
from torch import Tensor

def evaluate_simple(
    solution: Solution,
    explogger: Optional[ExperimentLogger] = None,
    **kwargs,
):
    """Evaluate a generated algorithm across dimensions, functions, and instances."""

    code_str = solution.code
    alg_name = solution.name

    try:
        ns, _ = prepare_namespace(
            code_str,
            allowed=[
                "torch",
                "gpytorch",
                "botorch",
                "scipy",
                "numpy",
                "math",
            ],
            logger=explogger,
        )

        # Use one namespace so generated methods can access imports
        # and helper functions defined in the generated code.
        exec(code_str, ns, ns)

        # Retrieve the generated algorithm directly from its namespace.
        algorithm_class = ns[alg_name]

        if not callable(algorithm_class):
            raise TypeError(f"{alg_name!r} is not callable.")

    except Exception as e:
        feedback = f"Error loading generated algorithm {alg_name}: {e}"
        solution.set_scores(-1.0, feedback)
        print(feedback)
        return solution

    dims = kwargs.get("dims", [2])
    multiplier = kwargs.get("multiplier", 30)
    reps = kwargs.get("reps", 3)

    aucs = []
    errors = []

    for dim in dims:
        budget = multiplier * dim

        l2 = AOCCLogger(
            budget=budget,
            upper=1e2,
            triggers=[ALWAYS],
            scale_log=True,
            stop_on_threshold=False,
        )

        for fid in [16]:
            for iid in [1]:
                problem = get_problem(
                    fid=fid,
                    instance=iid,
                    dimension=dim,
                )
                problem.attach_logger(l2)

                for rep in range(reps):
                    try:
                        algorithm = algorithm_class(
                            budget=budget,
                            random_seed=rep + 1000,
                            dim=dim,
                            minimize=True
                        )

                        algorithm(
                            problem

                        )

                    except OverBudgetException:
                        print(
                            f"Run exceeded budget for dim={dim}, "
                            f"fid={fid}, iid={iid}, rep={rep}."
                        )

                    except Exception as e:
                        message = (
                            f"Error running {alg_name} for dim={dim}, "
                            f"fid={fid}, iid={iid}, rep={rep}: {e}"
                        )
                        errors.append(message)
                        print(message)

                    try:
                        # Preserve your scoring policy: score the available
                        # trajectory even if execution stopped early.
                        auc = correct_aocc(problem, l2, budget)
                        aucs.append(float(auc))

                    finally:
                        l2.reset(problem)
                        problem.reset()

    # Aggregate only after every dimension has been evaluated.
    aucs = np.asarray(aucs, dtype=float)

    if aucs.size == 0 or not np.all(np.isfinite(aucs)):
        feedback = (
            f"Evaluation of {alg_name} produced no scores "
            "or contained non-finite AOCC values."
        )
        solution.set_scores(-1e-01, feedback)
        solution.add_metadata("evaluation_errors", errors)
        print(feedback)
        return solution

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))

    feedback = (
        f"The algorithm {alg_name} achieved an average Area over "
        f"the Convergence Curve (AOCC; 1.0 is best) of {auc_mean:.2f}, "
        f"with standard deviation {auc_std:.2f}."
    )

    if errors:
        feedback += (
            f"\nEncountered {len(errors)} execution error(s)."
            f"\nExample error: {errors[0]}"
        )

    print(feedback)

    solution.add_metadata("aucs", aucs.tolist())
    solution.add_metadata("evaluation_errors", errors)
    solution.set_scores(auc_mean, feedback)

    return solution

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 27.
from llamea import LLaMEA
import re, textwrap

# Denotes the role of the LLM in the optimization task, which is to generate Python code for Bayesian Optimization.
role_prompt = "You are an AI that generates Python optimization code by using Bayesian Optimization."

# The task prompt specifies the requirements for the generated code, including the class name, methods, and libraries to use.
# In this case we are prompting the LLM to generate a class that implements Bayesian Optimization using BoTorch and GPyTorch, with specific methods and functionality.
# We are asking a difficult task, which is to implement GP hedges which are difficult to implement and require a good understanding of the BoTorch framework 
# and the underlying mathematics.

task_prompt = textwrap.dedent("""\
Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that initializes the budget and dimension, 
the initial DoE points and the random seed (for ulterior Sobol samplers from torch or for LHS from scipy),
and a __call__(self, func) method that tries to find x in [-5, 5]^2 that minimizes func(x) (this can be set randomly, 
but imagine it's a highly multimodal function like the Gallagher 101 peaks).
The function should return the tuple best_f, best_x (value best found and the location).
Implement an efficient algorithm for this Bayesian Optimization task. The code you implement must
only use BoTorch and GPyTorch libraries for the optimization; 
You may also use scipy to generate random samples for the initial DoE and then transform the variables from lists or numpy arrays into torch tensors. 
Do not use any other optimization libraries. You may append additional helper functions or classes if needed, but the main class must be named Optimize_with_BO and must have the specified methods.
You can also use other BoTorch acquisition functions like qNoisyExpectedImprovement, qLogNoisyExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, and you can use the MaternKernel and ScaleKernel from GPyTorch.
You can also design your own acquisition function if you want, but it must be based on the BoTorch framework. Check for example the SAWEI from Benjamins et al. (2023) paper, which is a combination of qLogExpectedImprovement and qUpperConfidenceBound.
Or for instance take a look at the paper from HEBO and implement a GP-Hedge balancing the acquisition functions.
""")

# This is an example prompt that provides a template for the class that implements Bayesian Optimization using BoTorch and GPyTorch. 
# The user can fill in the implementation details for the methods as needed. As seen in previous sessions,
# we prompt the llm to implement the fallback for the _fit_model method to avoid the possibility of the fit_gpytorch_model function failing.

example_prompt = textwrap.dedent("""\
Here is an example of a class that implements Bayesian Optimization using BoTorch and GPyTorch:

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, minimize=True):
        self.budget = budget
        self.dim = dim
        if initial_points is None:
            self.initial_points = 5 * dim  # Default to 5 times the dimension if not provided
        else:
            self.initial_points = initial_points
        
        if random_seed is None:
            self.random_seed = 42  # Default random seed if not provided
        else:
            self.random_seed = random_seed
        
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.minimize = minimize

    def __call__(self, func):
        # Implementation for Bayesian Optimization
    
        
    def eval_function(self, x, func):
        # Implementation for evaluating the function at a given point x
        # This is implemented as a list comprehension that evaluates the function at each point in X and returns a tensor of the results.
        # The function always returns numpy and requires numpy arrays as input, 
        # so we detach the tensor and convert it to a numpy array before passing it to the function.
         y = torch.tensor([func(x.detach().cpu().numpy()) for x in X], dtype=torch.double)
        return y


    def _initial_design(self):
        # Implementation for generating initial design points
        pass
    
    def _fit_model(self, X, y):
        # Implementation for fitting a GP model
        # This is a fallback in case the fit_gpytorch_model function fails which must implement ADAM or other optimizer from torch.optim
        pass
        """)


es = LLaMEA(
    f=evaluate_simple,
    llm=gemini_llm_obj,
    n_parents=1, # implementing a (1,1) evolutionary strategy, which is a simple and effective approach for optimization tasks.
    n_offspring=1,
    role_prompt=role_prompt,
    task_prompt=task_prompt,
    example_prompt=example_prompt,
    experiment_name="myexperiment_BO_Hedge", # This is the name of the experiment, which will be used to save the results and logs.
    elitism=False,
    minimization=False, # The outer loop is a maximization (maximize the AOCC) while the inner loop is a minimization (minimize the function value).
    budget=5  # Try 5 iterations for a quick demo
)

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 28.
# Now run the evolutionary strategy to generate and evaluate the code for Bayesian Optimization.
best_solution = es.run()

print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
print(f"Generated code:\n{best_solution.code}")
print(f"Additional feedback: {best_solution.feedback}")

# TODO: The source files do not provide a complete configuration-driven runner for both assignment generation searches and persistent search-history output.
