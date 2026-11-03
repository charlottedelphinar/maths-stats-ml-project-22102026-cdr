# Online Shoppers: predicting purchases, models from scratch

## Project
- Course: Mathematics & Statistics for ML, Albert School
- Task: classification of `Revenue` (purchase yes or no)
- Data: Online Shoppers Purchasing Intention (Kaggle)
- Link: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset
- 12330 sessions, 15.47% purchases, 125 duplicate rows, no missing values
- Main model: weighted decision tree (Gini), NumPy only
- Other approaches: logistic regression, neural network (both from scratch)
- Benchmark: scikit-learn, same split, same metrics

## Method
- Split: seed 42, 60/20/20 (train/validation/test)
- Features: 16 variables, `PageValues` excluded
- Encoding: one-hot, columns fixed on train
- Imbalance: purchase weight, threshold
- Selection: validation only, test used once
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Selected models: plain, best F1, max precision with recall of at least 0.5

## Setup
- Python 3.13 or newer
- `python -m venv .venv` then `source .venv/bin/activate`
- `pip install -r requirements.txt`
- Data: download CSV from Kaggle, save as `data/online_shoppers_intention.csv`
- Not in git (see `.gitignore`)

## Run
- Everything: `bash run_experiments.sh` (about 6 minutes)
- Always from the project root, with `python -m`
- Tree: `python -m src.final_test`, `python -m src.visualize_results`
- Logistic regression: `python -m logistic_regression.logreg_scratch`
- Neural network: `python -m neural_network.nn_scratch`
- Comparison: `python -m compare_approaches`
- Experiment with PageValues: put `PAGE_VALUES=1` before any command
- Results end with `strict` or `pagevalues`

## Results (strict features, test set)
| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Decision tree, max precision | 0.346 | 0.538 | 0.421 |
| Logistic regression, max precision | 0.320 | 0.570 | 0.410 |
| Neural network, max precision | 0.312 | 0.559 | 0.401 |
- No approach reaches 0.5 for both metrics
- Limit: features, not algorithm
- PageValues: F1 about 0.63 to 0.64, leakage risk, labelled experiment
- Details: `test_result/README.md`

## Files
```
maths-stats-ml-project/
├── README.md                    this overview
├── requirements.txt             package versions
├── run_experiments.sh           runs everything
├── compare_approaches.py        tree vs logistic vs network
├── train_vs_test.py             train, validation, test gaps and split distribution
├── .gitignore
│
├── data/
│   └── online_shoppers_intention.csv     Kaggle file, not in git
│
├── src/                         main decision tree (current)
│   ├── eda.py                   exploration, target graph
│   ├── features.py              feature selection
│   ├── split_data.py            seed 42 split, PAGE_VALUES switch
│   ├── preprocessing.py         one-hot encoding
│   ├── tree_scratch.py          Gini, split, tree, predict, asserts
│   ├── run_project.py           validation search, 144 settings
│   ├── final_test.py            test once, scikit-learn benchmark
│   └── visualize_results.py     validation graphs
│
├── logistic_regression/
│   ├── logreg_scratch.py        sigmoid, gradient, descent, search
│   └── README.md
│
├── neural_network/
│   ├── nn_scratch.py            forward, backprop, search
│   └── README.md
│
├── old_decision_tree/           original tree, before changes
│   ├── (original code files)    runs as old_decision_tree.*
│   ├── old_visuals/             old graphs, renamed
│   ├── test_history/            4 test notebooks + README
│   └── README.md
│
├── results/                     current tree outputs
│   ├── validation_results_*.csv
│   ├── selected_models_*.csv
│   ├── test_results_*.csv
│   ├── model_comparison_*.png
│   ├── precision_recall_*.png
│   └── target_distribution.png
│
├── test_result/                 new approaches
│   ├── logistic_regression/     csv + 4 graphs
│   ├── neural_network/          csv + 4 graphs
│   ├── comparison/              three approaches side by side
│   └── README.md
│
├── notebooks/                   empty, for Colab notebooks
├── report/                      empty, for the PDF report
└── slides/                      empty, for the presentation
```

## Guidelines check
- From scratch: Gini, split search, recursion, gradients, backprop
- scikit-learn: benchmark only
- Fixed seeds, identical splits
- Leakage: scaling and encoding fitted on train
- Limits: one split, 125 duplicates, no confidence intervals
