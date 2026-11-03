import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

# Decision tree functions
from src.tree_scratch import build_tree, predict

# Getting the prepared datasets
from src.preprocessing import X_train_ready, X_val_ready
from src.split_data import y_train, y_val, TAG

# Converting targets to NumPy arrays
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Creating the results folder
Path("results").mkdir(exist_ok=True)

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

print()
print("Feature set:", TAG)

# Calculating the purchase weight using training data
purchase_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

# Listing the settings we want to try
depths = [3, 4, 5, 6, 7, 8]
min_leaves = [20, 50, 100]
purchase_weights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, float(purchase_ratio)]

# Comparing our tree with scikit-learn on the same settings
results = []

for depth in depths:
    for min_leaf in min_leaves:
        for purchase_weight in purchase_weights:

            # Giving more weight to purchases
            sample_weight = np.where(
                y_train == 1,
                purchase_weight,
                1.0
            )

            # Training our decision tree
            tree = build_tree(
                X_train_ready,
                y_train,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                sample_weight=sample_weight
            )

            # Evaluating our decision tree on validation data
            y_pred = predict(X_val_ready, tree)
            scores = classification_metrics(y_val, y_pred)

            results.append({
                "Model": "Our tree",
                "Depth": depth,
                "Min_leaf": min_leaf,
                "Purchase_weight": purchase_weight,
                **scores
            })

            # Training the scikit-learn decision tree
            sklearn_tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=42
            )

            sklearn_tree.fit(
                X_train_ready,
                y_train,
                sample_weight=sample_weight
            )

            # Evaluating the scikit-learn decision tree on validation data
            y_pred = sklearn_tree.predict(X_val_ready)
            scores = classification_metrics(y_val, y_pred)

            results.append({
                "Model": "Scikit-learn",
                "Depth": depth,
                "Min_leaf": min_leaf,
                "Purchase_weight": purchase_weight,
                **scores
            })

# Saving all validation results
table = pd.DataFrame(results)
table.to_csv(f"results/validation_results_{TAG}.csv", index=False)

print()
print("Best 5 settings by F1 on validation:")
print(table.sort_values("F1", ascending=False).head(5).round(4).to_string(index=False))

# Selecting the best model from our decision tree
our_results = table[table["Model"] == "Our tree"]

best_our_tree = our_results.loc[our_results["F1"].idxmax()]

# Selecting the best precision when recall stays at least 0.5
enough_recall = our_results[our_results["Recall"] >= 0.5]

if len(enough_recall) > 0:
    best_precision = enough_recall.loc[enough_recall["Precision"].idxmax()]
else:
    best_precision = best_our_tree

# Selecting the best tree with equal weights
equal_results = our_results[
    np.isclose(our_results["Purchase_weight"], 1.0)
]

best_equal = equal_results.loc[equal_results["F1"].idxmax()]

# Showing the three selected models
selected = pd.DataFrame([
    {"Method": "Equal weights", **best_equal.drop("Model")},
    {"Method": "Best F1", **best_our_tree.drop("Model")},
    {"Method": "Best precision with recall of at least 0.5", **best_precision.drop("Model")},
])

selected.to_csv(f"results/selected_models_{TAG}.csv", index=False)

print()
print("Selected models on validation:")
print(selected.round(4).to_string(index=False))

# Comparing with the baseline that never predicts a purchase
baseline = np.zeros(len(y_val), dtype=int)
baseline_scores = classification_metrics(y_val, baseline)

print()
print("Baseline results:")
print(pd.Series(baseline_scores).round(4))

# Checkpoints
assert len(table) == 2 * len(depths) * len(min_leaves) * len(purchase_weights)
assert table["F1"].between(0, 1).all()
assert table["Accuracy"].between(0, 1).all()
assert np.isclose(baseline_scores["Recall"], 0.0)

print()
print("OK - model comparison completed")
