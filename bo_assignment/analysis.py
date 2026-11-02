# PURPOSE AND REMAINING WORK
#
# This module should turn stored experiment output into the evidence required
# by the assignment. It should load raw per-seed results, align convergence
# curves, calculate means and variability, compare original-target and transfer
# performance, and create the LLaMEA AOCC and Stage 2 convergence plots.
#
# The copied benchmarking material provides IOH data-management imports and
# plotting/data-loading examples. These blocks show how course experiment data
# can be cleaned, loaded, aligned, and plotted, but they are not yet connected
# to the new JSON/CSV result layout or to the assignment's four methods and two
# problems.
#
# Still missing:
# - the agreed raw-result schema and loader;
# - aggregation grouped by method, problem, and seed;
# - best-so-far curves with variability bands;
# - LLaMEA outer-loop AOCC plots;
# - final tables and specialization/generalization comparisons;
# - output paths and reproducible figure generation.

# This section combines the AOCC calculation with result analysis so that one
# module owns both the measurements and their interpretation. The blocks below
# remain copied from the original LLaMEA working-group notebook.

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 8.
# Define a custom logger class that inherits from AbstractLogger
# Extracted from: https://github.com/XAI-liacs/LLaMEA/blob/main/misc/utils.py
class AOCCLogger(AbstractLogger):
    """AOCC_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        lower=1e-8,
        upper=1e8,
        scale_log=False,
        stop_on_threshold=False,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.aocc = 0.0
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.stop_on_threshold = stop_on_threshold
        self.transform = lambda x: np.log10(x) if scale_log else x

    def __call__(self, log_info: LogInfo):
        """Subcalculate the AOCC.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations > self.budget:
            raise OverBudgetException
        if log_info.evaluations == self.budget:
            return
        if self.stop_on_threshold and abs(log_info.raw_y_best) < self.lower:
            raise ThresholdReachedException
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper)
        self.aocc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )

    def reset(self, func):
        super().reset()
        self.aocc = 0

# Copied verbatim from Relevent Files/LLaMEA_Working_Groups_students.ipynb, code cell 10.
# This same function is found at: https://github.com/XAI-liacs/LLaMEA/blob/main/misc/utils.py
def correct_aocc(ioh_function, logger, budget):
    """Correct aocc values in case a run stopped before the budget was exhausted

    Args:
        ioh_function: The function in its final state (before resetting!)
        logger: The logger in its final state, so we can ensure the settings for aocc calculation match
        budget: The intended maximum budget

    Returns:
        float: The normalized aocc of the run, corrected for stopped runs
    """
    fraction = (
        logger.transform(
            np.clip(
                ioh_function.state.current_best_internal.y, logger.lower, logger.upper
            )
        )
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper) - logger.transform(logger.lower))
    aocc = (
        logger.aocc
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    return 1 - aocc

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 23.
# This is a prelude to manage the files that will be generated during this exercise
import os
import glob

from shutil import rmtree

def clean():
    for name in ("my-experiment", "ioh_data"):
        for path in glob.glob(f"{name}*"):
            if os.path.isfile(path):
                os.remove(path)
            if os.path.isdir(path):
                rmtree(path, ignore_errors=True)

def ls(p="./"):
    for obj in os.listdir(os.path.normpath(p)):
        print(obj)

def cat(f):
    with open(os.path.normpath(f)) as h:
        print(h.read())

clean()

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

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 41.
manager = DataManager()
manager.add_folder("my-experiment_Genetic Algorithm")
manager.add_folder("my-experiment_Differential Evolution")
manager.add_folder("my-experiment_Bayesian Optimization")

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 44.
# Load the performance data
df = manager.load(monotonic=True,
                  include_meta_data=True)

# Align the data (you can also use the function `align_data` from iohinspector)
df = turbo_align(df,x_values=[*range(1,103,5)],x_col="evaluations",y_col="raw_y", maximization=False)

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 47.
# This block is to make a plot of each run of the experiment
fig, ax = plt.subplots(figsize=(10, 7))
linestyles = ['-', '-.', ':']
colors = ['blue', 'red', 'green', 'magenta', 'cyan']

for idx_alg, algorithm in enumerate(df['algorithm_name'].unique()):
    # Filter the dataframe for the current algorithm
    df_algo = df.filter(pl.col('algorithm_name') == algorithm)
    
    for idx_run, run in enumerate(df_algo['run_id'].unique()):

        # Filter the dataframe for the current run
        df_run = df_algo.filter(pl.col('run_id') == run)
        ax.plot(df_run['evaluations'], df_run['raw_y_best'], 
                label=f'Run {run}_{algorithm}',
                linestyle=linestyles[idx_alg],
                color = colors[idx_run])

ax.grid(True)
ax.set_xlabel('Function Evaluations')
ax.set_ylabel('Regret')
ax.set_yscale('log')
ax.set_title('Convergence Plot per Run')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='xx-small')
plt.tight_layout()  # Adjust layout to prevent clipping)

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 49.
# Convergence plot: mean ± SE across runs for each algorithm
fig, ax = plt.subplots(figsize=(10, 7))

for algorithm in df.get_column('algorithm_name').unique().to_list():
    # Filter this algorithm
    df_algo = df.filter(pl.col('algorithm_name') == algorithm)

    # Aggregate stats per evaluation count
    df_stats = (
        df_algo
        .group_by('evaluations')
        .agg(
            mean_regret = pl.col('raw_y_best').mean(),
            std_regret  = pl.col('raw_y_best').std(),     # sample std (ddof=1)
            n           = pl.len()
        )
        .with_columns(
            se_regret = pl.col('std_regret') / pl.col('n').sqrt()
        )
        .select(['evaluations', 'mean_regret', 'se_regret'])
        .sort('evaluations')
    )

    x = df_stats['evaluations'].to_numpy()
    m = df_stats['mean_regret'].to_numpy()
    se = df_stats['se_regret'].to_numpy()*1.2 # 95% Confidence Interval

    ax.plot(x, m, label=str(algorithm), linestyle='--', linewidth=2)
    ax.fill_between(x, m - se, m + se, alpha=0.2, linewidth=0)

ax.grid(True, which='both', alpha=0.3)
ax.set_xlabel('Function Evaluations')
ax.set_ylabel('Mean Regret')
ax.set_yscale('log')  # ensure regrets are strictly > 0
ax.set_title('Convergence (mean ± SE)')

# Place legend outside the plot
leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='xx-small', frameon=False)
plt.tight_layout()

# Copied verbatim from Relevent Files/Benchmarking_Solutions_2026.ipynb, code cell 51.
# Modify the plot to have the Standard error

from iohinspector.plot import single_function_fixedbudget
fig, ax = plt.subplots(figsize=(10, 7))
data_singlefb = single_function_fixedbudget(df, maximization=False, measures=["mean"], ax=ax, scale_xlog=False)

# Modification of the legend
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
ax.invert_xaxis() # This is used since the default behavior is to have the x-axis in decreasing order
ax.set_title('Fixed-Budget Plot - Mean Regret')

# TODO: The source files do not provide a complete reusable result loader and analysis pipeline for the assignment output format.
