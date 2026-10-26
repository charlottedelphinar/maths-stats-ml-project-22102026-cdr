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

# Calculating the purchase weight using training data
purchase_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

# Comparing different purchase weights
purchase_weights = [1.0, 2.0, 3.0, 5.0, float(purchase_ratio)]

for depth in [3, 5, 7]:
    for purchase_weight in purchase_weights:

        # Giving more weight to purchases
        sample_weight = np.where(
            y_train == 1,
            purchase_weight,
            1.0
        )

        # Training decision tree
        tree = build_tree(
            X_train_ready,
            y_train,
            max_depth=depth,
            min_samples_leaf=20,
            sample_weight=sample_weight
        )

        # Evaluating decision tree
        y_pred = predict(X_val_ready, tree)
        scores = classification_metrics(y_val, y_pred)

        results.append({
            "Model": "Our tree",
            "Depth": depth,
            "Purchase_weight": purchase_weight,
            **scores
        })

        # Training scikit-learn decision tree
        sklearn_tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_leaf=20,
            random_state=42
        )

        sklearn_tree.fit(
            X_train_ready,
            y_train,
            sample_weight=sample_weight
        )

        # Evaluating scikit-learn decision tree
        y_pred = sklearn_tree.predict(X_val_ready)
        scores = classification_metrics(y_val, y_pred)

        results.append({
            "Model": "Scikit-learn",
            "Depth": depth,
            "Purchase_weight": purchase_weight,
            **scores
        })


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
print("Best purchase weight:", round(best["Purchase_weight"], 4))
print("Best F1-score:", round(best["F1"], 4))


# Selecting best model from our decision tree
our_results = table[table["Model"] == "Our tree"]

best_our_tree = our_results.loc[
    our_results["F1"].idxmax()
]

print()
print("Best model from scratch:")
print("Best depth:", int(best_our_tree["Depth"]))
print("Best purchase weight:", round(best_our_tree["Purchase_weight"], 4))
print("Best F1-score:", round(best_our_tree["F1"], 4))


# Comparing equal weights with weighted purchases
equal_results = our_results[
    np.isclose(our_results["Purchase_weight"], 1.0)
]

best_equal = equal_results.loc[
    equal_results["F1"].idxmax()
]

comparison = pd.DataFrame([
    {
        "Method": "Equal weights",
        "Depth": int(best_equal["Depth"]),
        "Purchase_weight": 1.0,
        "Accuracy": best_equal["Accuracy"],
        "Precision": best_equal["Precision"],
        "Recall": best_equal["Recall"],
        "F1": best_equal["F1"]
    },
    {
        "Method": "Weighted purchases",
        "Depth": int(best_our_tree["Depth"]),
        "Purchase_weight": best_our_tree["Purchase_weight"],
        "Accuracy": best_our_tree["Accuracy"],
        "Precision": best_our_tree["Precision"],
        "Recall": best_our_tree["Recall"],
        "F1": best_our_tree["F1"]
    }
])

print()
print("Equal weights vs weighted purchases:")
print(comparison.round(4).to_string(index=False))


# Comparing with baseline (?) check again!
baseline = np.zeros(len(y_val), dtype=int)
baseline_scores = classification_metrics(y_val, baseline)

print()
print("Baseline results:")
print(pd.Series(baseline_scores).round(4))


# Checkpoints
assert len(table) == 30
assert table["F1"].between(0, 1).all()
assert table["Accuracy"].between(0, 1).all()
assert np.isclose(baseline_scores["Recall"], 0.0)

print()
print("OK - model comparison completed")
