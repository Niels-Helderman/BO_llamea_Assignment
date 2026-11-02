# PURPOSE AND REMAINING WORK
#
# This module should be the single place responsible for creating and exposing
# the two assignment problems: the IOH/COCO-BBOB problem and the practical
# real-world problem. It should eventually contain a shared problem interface
# or factory, problem-specific construction functions, dimension and instance
# handling, objective-direction handling, reset handling, and any common
# evaluation-budget behaviour.
#
# The copied imports and exceptions below provide pieces needed by that work.
# The random_search_alg block shows how an IOH RealSingleObjective exposes
# bounds and its number of variables. The get_practical_problem import shows
# the entry point supplied for the practical-problem suite. Together, these
# blocks can be used as the basis for problem constructors, but they do not
# yet provide one consistent interface for the rest of the assignment.
#
# Still missing:
# - configuration-driven BBOB and real-world problem factories;
# - a common way to obtain bounds, dimension, and problem identifiers;
# - consistent reset and evaluation-count handling;
# - the exact assigned problem settings;
# - validation that both problem types follow the same minimization contract.

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

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 6.
class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""
    pass

class ThresholdReachedException(Exception):
    """The algorithm reached the threshold of the problem."""
    pass

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

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 15.
from Practical_Problems import get_practical_problem

# TODO: The source files do not provide a shared problem-wrapper API or a complete assignment problem factory.
