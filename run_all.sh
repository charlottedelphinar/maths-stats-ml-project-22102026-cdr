#!/bin/bash
# Running the whole project from the project root
set -e

# Moving to the folder where this script is saved
cd "$(dirname "$0")"

# Activating the virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Checking our tree with the small tests
python -m src.tree_scratch

# Running the main result without PageValues
echo
echo "===== Strict features ====="
python -m src.final_test
python -m src.visualize_results

# Running the experiment with PageValues
echo
echo "===== With PageValues ====="
PAGE_VALUES=1 python -m src.final_test
PAGE_VALUES=1 python -m src.visualize_results

echo
echo "Everything finished, see the results folder"
