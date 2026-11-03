#!/bin/bash
# Running the logistic regression and neural network experiments from the project root
set -e

# Moving to the folder where this script is saved
cd "$(dirname "$0")"

# Activating the virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Running the strict feature set, this is the main result
echo "===== Strict features ====="
python -m src.final_test
python -m logistic_regression.logreg_scratch
python -m neural_network.nn_scratch

# Running the experiment with PageValues
echo
echo "===== With PageValues ====="
PAGE_VALUES=1 python -m src.final_test
PAGE_VALUES=1 python -m logistic_regression.logreg_scratch
PAGE_VALUES=1 python -m neural_network.nn_scratch

# Comparing the three approaches
python -m compare_approaches

echo
echo "Everything finished, see the test_result folder"
