import numpy as np

# Getting our decision tree functions
from src.tree_scratch import build_tree, predict
from src.run_project import classification_metrics

# Getting the prepared datasets
from src.preprocessing import X_train_ready, X_test_ready
from src.split_data import y_train, y_test

# Converting targets to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Training the selected decision tree
tree = build_tree(
    X_train_ready,
    y_train,
    max_depth=7,
    min_samples_leaf=20
)

# Predicting test data
y_pred = predict(X_test_ready, tree)

# Calculating test results
scores = classification_metrics(y_test, y_pred)

print()
print("Final test results:")
print("Accuracy:", round(scores["Accuracy"], 4))
print("Precision:", round(scores["Precision"], 4))
print("Recall:", round(scores["Recall"], 4))
print("F1-score:", round(scores["F1"], 4))

# Comparing with baseline
baseline = np.zeros(len(y_test), dtype=int)
baseline_scores = classification_metrics(y_test, baseline)

print()
print("Baseline test results:")
print("Accuracy:", round(baseline_scores["Accuracy"], 4))
print("Precision:", round(baseline_scores["Precision"], 4))
print("Recall:", round(baseline_scores["Recall"], 4))
print("F1-score:", round(baseline_scores["F1"], 4))

# Checkpoints
assert len(y_pred) == len(y_test)
assert set(np.unique(y_pred)).issubset({0, 1})

print()
print("OK - final test completed")