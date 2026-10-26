import numpy as np
import pandas as pd

# Getting our decision tree functions
from src.tree_scratch import build_tree, predict
from src.run_project import classification_metrics, best_equal, best_our_tree

# Getting the prepared datasets
from src.preprocessing import X_train_ready, X_test_ready
from src.split_data import y_train, y_test

# Converting targets to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# Training the selected decision tree
def test_model(name, depth, purchase_weight):
    sample_weight = np.where(
        y_train == 1,
        purchase_weight,
        1.0
    )

    tree = build_tree(
        X_train_ready,
        y_train,
        max_depth=int(depth),
        min_samples_leaf=20,
        sample_weight=sample_weight
    )

    # Predicting test data
    y_pred = predict(X_test_ready, tree)

    # Calculating test results
    scores = classification_metrics(y_test, y_pred)

    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    print()
    print("Final test results:", name)
    print("Depth:", int(depth))
    print("Purchase weight:", round(purchase_weight, 4))
    print("Accuracy:", round(scores["Accuracy"], 4))
    print("Precision:", round(scores["Precision"], 4))
    print("Recall:", round(scores["Recall"], 4))
    print("F1-score:", round(scores["F1"], 4))

    print()
    print("Confusion matrix:")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    # Checkpoints
    assert len(y_pred) == len(y_test)
    assert set(np.unique(y_pred)).issubset({0, 1})

    return {
        "Model": name,
        "Depth": int(depth),
        "Purchase_weight": purchase_weight,
        **scores
    }


# Comparing equal weights with weighted purchases
results = [
    test_model(
        "Equal weights",
        best_equal["Depth"],
        1.0
    ),
    test_model(
        "Weighted purchases",
        best_our_tree["Depth"],
        best_our_tree["Purchase_weight"]
    )
]

# Displaying comparison results
comparison = pd.DataFrame(results)

print()
print("Final test comparison:")
print(comparison.round(4).to_string(index=False))


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
assert len(comparison) == 2
assert comparison["F1"].between(0, 1).all()
assert comparison["Accuracy"].between(0, 1).all()

print()
print("OK - final test completed")