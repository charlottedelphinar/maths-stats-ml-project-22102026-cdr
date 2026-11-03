import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Getting our decision tree functions
from src.tree_scratch import build_tree, predict
from src.run_project import classification_metrics, best_equal, best_our_tree, best_precision

# Getting the prepared datasets
from src.preprocessing import X_train_ready, X_test_ready
from src.split_data import y_train, y_test, TAG

# Converting targets to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Training one model and testing it once
def test_model(name, row):
    depth = int(row["Depth"])
    min_leaf = int(row["Min_leaf"])
    purchase_weight = float(row["Purchase_weight"])

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
    print("Depth:", depth, "| Min leaf:", min_leaf, "| Purchase weight:", round(purchase_weight, 4))
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

    # Training scikit-learn with the same settings
    sklearn_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_leaf=min_leaf,
        random_state=42
    )

    sklearn_tree.fit(X_train_ready, y_train, sample_weight=sample_weight)
    sk_scores = classification_metrics(y_test, sklearn_tree.predict(X_test_ready))

    return [
        {"Model": "Our tree", "Method": name, "Depth": depth, "Min_leaf": min_leaf,
         "Purchase_weight": purchase_weight, "TP": tp, "FP": fp, "FN": fn, "TN": tn, **scores},
        {"Model": "Scikit-learn", "Method": name, "Depth": depth, "Min_leaf": min_leaf,
         "Purchase_weight": purchase_weight, **sk_scores},
    ]

# Testing the three selected models once
results = []
results += test_model("Equal weights", best_equal)
results += test_model("Best F1", best_our_tree)
results += test_model("Best precision with recall of at least 0.5", best_precision)

# Displaying and saving the comparison
comparison = pd.DataFrame(results)
comparison.to_csv(f"results/test_results_{TAG}.csv", index=False)

print()
print("Final test comparison:")
print(comparison[["Model", "Method", "Depth", "Min_leaf", "Purchase_weight",
                  "Accuracy", "Precision", "Recall", "F1"]].round(4).to_string(index=False))

# Comparing with the baseline that never predicts a purchase
baseline = np.zeros(len(y_test), dtype=int)
baseline_scores = classification_metrics(y_test, baseline)

print()
print("Baseline test results:")
print("Accuracy:", round(baseline_scores["Accuracy"], 4))
print("Precision:", round(baseline_scores["Precision"], 4))
print("Recall:", round(baseline_scores["Recall"], 4))
print("F1-score:", round(baseline_scores["F1"], 4))

# Checkpoints
assert len(comparison) == 6
assert comparison["F1"].between(0, 1).all()
assert comparison["Accuracy"].between(0, 1).all()

print()
print("OK - final test completed")
