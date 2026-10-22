import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Decision tree functions
from src.tree_scratch import build_tree, predict

# Getting the prepared datasets
from src.preprocessing import X_train_ready, X_val_ready
from src.split_data import y_train, y_val

# Converting targets to NumPy arrays
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Calculating classification metrics
def classification_metrics(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    accuracy = np.mean(y == y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

# Comparing tree with scikit-learn
results = []

for depth in [3, 5, 7]:

    # Training decision tree
    tree = build_tree(
        X_train_ready, y_train,
        max_depth=depth,
        min_samples_leaf=20
    )

    # Evaluating decision tree
    y_pred = predict(X_val_ready, tree)
    scores = classification_metrics(y_val, y_pred)
    results.append({"Model": "Our tree", "Depth": depth, **scores})

    # Training scikit-learn decision tree
    sklearn_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_leaf=20,
        random_state=42
    )
    sklearn_tree.fit(X_train_ready, y_train)

    # Evaluating scikit-learn decision tree
    y_pred = sklearn_tree.predict(X_val_ready)
    scores = classification_metrics(y_val, y_pred)
    results.append({"Model": "Scikit-learn", "Depth": depth, **scores})

# Displaying comparison results
table = pd.DataFrame(results)

print()
print("Validation results:")
print(table.round(4).to_string(index=False))

# Selecting best model
best = table.loc[table["F1"].idxmax()]

print()
print("Best model:", best["Model"])
print("Best depth:", int(best["Depth"]))
print("Best F1-score:", round(best["F1"], 4))

# Comparing with baseline (?) check again!
baseline = np.zeros(len(y_val), dtype=int)
baseline_scores = classification_metrics(y_val, baseline)

print()
print("Baseline results:")
print(pd.Series(baseline_scores).round(4))

# Checkpoints
assert len(table) == 6
assert table["F1"].between(0, 1).all()
assert table["Accuracy"].between(0, 1).all()
assert np.isclose(baseline_scores["Recall"], 0.0)

print()
print("OK - model comparison completed")