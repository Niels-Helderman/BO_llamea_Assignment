# PURPOSE AND REMAINING WORK
#
# This script should be the user-facing entry point for Stage 1. It should
# read the assignment and prompt configuration, run LLaMEA once for the BBOB
# target and once for the real-world target, and save the selected algorithms,
# prompts, scores, logs, and generation history under artifacts/ and results/.
#
# The copied LLaMEA setup and evaluator in bo_assignment/generation.py are
# the implementation material this script should call. No existing source file
# contains command-line parsing, configuration loading, or the two-target
# orchestration, so those parts still need to be implemented.
