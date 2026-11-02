# PURPOSE AND REMAINING WORK
#
# This module should orchestrate the assignment benchmark without containing
# problem-specific notebook loops. It should consume the two Stage 1 generated
# outputs and provide the Stage 2 benchmark matrix: both generated algorithms,
# CMA-ES, and BoTorch evaluated on both target problems with the same dimension,
# budgets, seeds, initialization accounting, and minimization convention.
#
# The copied AOCC loops demonstrate attaching an AOCC logger, running repeated
# IOH/practical-problem evaluations, correcting incomplete runs, resetting the
# logger and problem, and aggregating scores. They can be used as the pattern
# for the Stage 1 evaluator, but they use demonstration dimensions, function
# IDs, and repetition counts rather than the final assignment configuration.
# The copied Benchmarking notebook imports and IOH logging setup provide the
# data and logging dependencies used by the course's benchmark examples.
#
# Still missing:
# - reusable Stage 2 benchmark functions;
# - configuration-driven loops instead of hard-coded notebook values;
# - the full four-method-by-two-problem benchmark matrix;
# - raw result serialization for every method and seed;
# - consistent failure handling and run metadata;
# - separation between generation-time budget 60 and benchmark budget 600.

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

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 24.
# Import the IOH library
# pip install ioh
import ioh

# Import numpy ## For Numerical Processing
import numpy as np

# Import pyplot ## For plotting purposes
import matplotlib.pyplot as plt

# Import the iohinspector Data Manage, the function `turbo_align` and polars
from iohinspector import DataManager
from iohinspector import turbo_align

import polars as pl


# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 13.
# Store the AUCs for each run
aucs = []

for dim in [5]:
    budget = 30 * dim
    l2 = AOCCLogger(budget=budget, upper=1e2, triggers=[ALWAYS], scale_log=False, stop_on_threshold=False)

    for fid in np.arange(16, 25):
        for iid in [1, 2, 3]:  
            problem = get_problem(fid=fid, instance=iid, dimension=dim)
            problem.attach_logger(l2)

            for rep in range(3):
                try:
                    algorithm = random_search_alg
                    algorithm(problem, budget=budget, seed=rep+1000, dimension=dim, minimize=True)
                except OverBudgetException:
                    print(f"Run exceeded budget for fid={fid}, iid={iid}, rep={rep}.")
                    pass

                auc = correct_aocc(problem, l2, budget)
                aucs.append(auc)
                l2.reset(problem)
                problem.reset()
    aucs = np.array(aucs)
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    print(f"Dimension: {dim}, Mean AUC: {auc_mean:.4f}, Std AUC: {auc_std:.4f}")
    print(f"All AUCs: {aucs}")

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 16.
aucs = []

for dim in [6]:
    budget = 30 * dim
    l2 = AOCCLogger(budget=budget, upper=1e10, lower=1e-10, triggers=[ALWAYS], scale_log=False, stop_on_threshold=False)

    for fid in [1124, 1125, 1129]:
    #for fid in [1130]: # Activate if you want to test, however this one is slow.

        if fid==1124:
            l2.scale_log = True
            l2.lower = 1e-3
            l2.upper = 1e3
        elif fid==1125:
            l2.lower = -5.00
            l2.upper = 1e-5
        elif fid==1129:
            l2.lower = -10.0
            l2.upper = 1e-08

        elif fid==1130:
            l2.lower = -1e11
            l2.upper = 1e-08
        problem = get_practical_problem(idx=fid, dim=dim)
        problem.attach_logger(l2)

        for rep in range(3):
            try:
                algorithm = random_search_alg
                algorithm(problem, budget=budget, seed=rep+1000, dimension=dim, minimize=True)
            except OverBudgetException:
                print(f"Run exceeded budget for fid={fid}, iid={iid}, rep={rep}.")
                pass

            auc = correct_aocc(problem, l2, budget)
            aucs.append(auc)
            l2.reset(problem)
            problem.reset()
    aucs = np.array(aucs)
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    print(f"Dimension: {dim}, Mean AOCC: {auc_mean:.4f}, Std AOCC: {auc_std:.4f}")
    print(f"All AOCCs: {aucs}")

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 36.

n_runs = 3
budget = 100

for algorithm_class, problem in zip([GeneticAlgorithm, DifferentialEvolution, BayesianOptimization],[rosenbrock_2d_problem]*3):

    # Initialize the algorithm with the problem dimension and bounds
    algorithm = algorithm_class(dim=problem.meta_data.n_variables,
                                lb = problem.bounds.lb,
                                ub = problem.bounds.ub
                                )
    # IOH implements a reset method to restart the problem state
    problem.reset()
    
    print(f"\nAlgorithm: {algorithm.name} - Function: {problem.meta_data.name}  - Instance: {problem.meta_data.instance} - Dimension: {problem.meta_data.n_variables}")
    for run in range(n_runs):

        # Run the algorithm
        algorithm.run(problem,budget=budget, random_seed=run+45)

        # print the best found for this run
        print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}"\
              f" - regret:{abs(problem.state.current_best.y-problem.optimum.y): .3f}"\
              f" - num evals: {problem.state.evaluations}")
        problem.reset()

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 38.
n_runs = 5
budget = 100

# Presets for logging

# Set the trigger characteristics (when to save evaluations done by the algorithm)
triggers = [
    ioh.logger.trigger.Each(5), # Log every (n) evaluations
    ioh.logger.trigger.OnImprovement() # On every improvement
]

# Set the properties to be logged (what to save)
additional_properties = [ioh.logger.property.CURRENTY,  # This is to log the current function value
                         ioh.logger.property.RAWYBEST, # This is to log the best-so-far regret
                         ioh.logger.property.CURRENTBESTY] # This is to log the best-so-far function value

# Root directory to store the data
root_directory = os.getcwd()

# Base Folder name to store the data
base_folder_name = "my-experiment"

# This is to store the (design) points evaluated during the optimization
store_positions = True 


for algorithm_class, problem in zip([GeneticAlgorithm, DifferentialEvolution, BayesianOptimization],[rosenbrock_2d_problem]*3):

    # Initialize the algorithm with the problem dimension and bounds
    algorithm = algorithm_class(dim=problem.meta_data.n_variables,
                                lb = problem.bounds.lb,
                                ub = problem.bounds.ub
                                )
    
    # Restart the problem state
    problem.reset()

    # Initialize the logger
    logger = ioh.logger.Analyzer(
        root=root_directory,
        folder_name=base_folder_name+f"_{algorithm.name}",
        algorithm_name=algorithm.name,
        additional_properties=additional_properties,
        triggers=triggers,
        store_positions=store_positions
    )

    # Attach the logger to the problem
    problem.attach_logger(logger)
    
    
    print(f"\nAlgorithm: {algorithm.name} - Function: {problem.meta_data.name}  - Instance: {problem.meta_data.instance} - Dimension: {problem.meta_data.n_variables}")
    for run in range(n_runs):

        # Run the algorithm
        algorithm.run(problem,budget=budget, random_seed=run+145)

        # print Evaluation Finished
        print(f"run: {run+1} Finished")
        problem.reset()
    
    # Detach the logger from the problem
    problem.detach_logger()

    # Close the logger
    logger.close()

# TODO: The source files do not provide the complete Stage 1/Stage 2 experiment orchestration for all four methods on both assignment problems.
