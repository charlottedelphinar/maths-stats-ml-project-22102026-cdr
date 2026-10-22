import numpy as np

# Calculating Gini impurity
def gini(y):
    if len(y) == 0:
        return 0.0

    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

# Calculating weighted Gini impurity
def weighted_gini(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right

    if n == 0:
        return 0.0

    return (n_left / n) * gini(y_left) + (n_right / n) * gini(y_right)

# Finding the best feature and threshold
def best_split(X, y, min_samples_leaf=20):
    best_feature = None
    best_threshold = None
    best_score = gini(y)

    for j in range(X.shape[1]):
        values = np.unique(X[:, j])

        if len(values) <= 1:
            continue

        if len(values) <= 16:
            thresholds = (values[:-1] + values[1:]) / 2
        else:
            thresholds = np.unique(
                np.quantile(values, np.linspace(0, 1, 17)[1:-1])
            )

        for t in thresholds:
            left = X[:, j] <= t
            right = ~left

            if left.sum() < min_samples_leaf or right.sum() < min_samples_leaf:
                continue

            score = weighted_gini(y[left], y[right])

            if score < best_score - 1e-12:
                best_score = score
                best_feature = j
                best_threshold = t

    return best_feature, best_threshold

# Building the decision tree
def build_tree(X, y, depth=0, max_depth=5, min_samples_leaf=20):
    prediction = int(np.mean(y) >= 0.5)

    if depth >= max_depth or gini(y) == 0:
        return {"prediction": prediction}

    j, t = best_split(X, y, min_samples_leaf)

    if j is None:
        return {"prediction": prediction}

    left = X[:, j] <= t
    right = ~left

    return {
        "feature": j,
        "threshold": t,
        "left": build_tree(X[left], y[left], depth + 1, max_depth, min_samples_leaf),
        "right": build_tree(X[right], y[right], depth + 1, max_depth, min_samples_leaf),
    }

# Predicting one shopping session
def predict_one(x, tree):
    if "prediction" in tree:
        return tree["prediction"]

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(x, tree["left"])

    return predict_one(x, tree["right"])

# Predicting all shopping sessions
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# Calculating classification results
def evaluate(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    accuracy = np.mean(y == y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

# Checkpoints and training
if __name__ == "__main__":

    # Check Gini impurity
    assert np.isclose(gini(np.array([0, 0, 0, 0])), 0.0)
    assert np.isclose(gini(np.array([0, 0, 1, 1])), 0.5)
    assert np.isclose(gini(np.array([0, 0, 0, 0, 1])), 0.32)
    print("OK - Gini impurity")

    # Check weighted Gini impurity
    assert np.isclose(weighted_gini(np.array([0, 0]), np.array([1, 1])), 0.0)
    assert np.isclose(weighted_gini(np.array([0, 1]), np.array([0, 1])), 0.5)
    assert np.isclose(weighted_gini(np.array([0, 0, 1]), np.array([1])), 1 / 3)
    print("OK - weighted Gini impurity")

    # Check the tree on a small dataset
    X_small = np.array([[1], [2], [3], [4]], dtype=float)
    y_small = np.array([0, 0, 1, 1])

    j, t = best_split(X_small, y_small, min_samples_leaf=1)
    assert j == 0
    assert np.isclose(t, 2.5)
    print("OK - best split")

    small_tree = build_tree(
        X_small, y_small, max_depth=2, min_samples_leaf=1
    )
    assert np.array_equal(predict(X_small, small_tree), y_small)
    print("OK - tree predictions")

    # Getting the prepared shopping data
    from src.preprocessing import X_train_ready, X_val_ready
    from src.split_data import y_train, y_val

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    # Training the decision tree
    tree = build_tree(
        X_train_ready,
        y_train,
        max_depth=5,
        min_samples_leaf=20,
    )

    print()
    print("OK - decision tree trained")

    # Making predictions on validation data
    y_pred = predict(X_val_ready, tree)

    print()
    print("Validation results:")
    evaluate(y_val, y_pred)

    # Comparing with a no-purchase baseline
    baseline = np.zeros(len(y_val), dtype=int)

    print()
    print("Baseline results:")
    evaluate(y_val, baseline)