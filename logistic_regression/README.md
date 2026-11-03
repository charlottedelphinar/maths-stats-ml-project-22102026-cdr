# Logistic regression from scratch

One script: `logreg_scratch.py`. It follows Lab 1, Part B (logistic regression from scratch).

## What it does
1. Standardising the features with the training mean only.
2. Writing the sigmoid, the cross entropy and the gradient ourselves.
3. Checking the gradient with finite differences and asserts.
4. Training with our own gradient descent.
5. Searching purchase weights, ridge strength and thresholds on validation data.
6. Testing the selected models once and comparing with scikit-learn.

## How to run
From the project root:

    python -m logistic_regression.logreg_scratch
    PAGE_VALUES=1 python -m logistic_regression.logreg_scratch

## What we select
- Plain logistic regression with threshold 0.5.
- Best F1 on validation.
- Max precision on validation, with recall of at least 0.5.
  We need the recall condition because precision alone can be pushed to 1.0 by predicting
  only a handful of sessions (see the threshold graph).

Results are saved in `test_result/logistic_regression/`.
