# Test results of the new approaches

Every file name ends with `strict` (16 features, our main result) or `pagevalues` (experiment with PageValues added).

| Folder | Made by | Contents |
|---|---|---|
| `logistic_regression/` | `python -m logistic_regression.logreg_scratch` | validation grid, selected models, test results, 4 graphs |
| `neural_network/` | `python -m neural_network.nn_scratch` | validation grid, selected models, test results, 4 graphs |
| `comparison/` | `python -m compare_approaches`, `python -m train_vs_test` | three approaches side by side, confusion matrices, train vs test gaps, split distribution |

The decision tree results stay in `results/` (made by `src/`). The older tree versions are in `old_decision_tree/`.

## What we select, for every approach
All choices use validation data only. The test set is used once at the end.
1. Plain model (equal weights, threshold 0.5)
2. Best F1
3. Max precision with a recall of at least 0.5

## Main finding (strict features, test set)
| Approach | Max precision with recall of at least 0.5 | Best F1 |
|---|---|---|
| Decision tree | precision 0.346, recall 0.538, F1 0.421 | F1 0.421 |
| Logistic regression | precision 0.320, recall 0.570, F1 0.410 | F1 0.389 |
| Neural network | precision 0.312, recall 0.559, F1 0.401 | F1 0.390 |

None of the three reaches 0.5 for both precision and recall with the 16 strict features.
The weighted decision tree is slightly ahead, so changing the algorithm does not fix the problem.
