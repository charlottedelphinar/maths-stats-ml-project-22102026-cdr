import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Getting the prepared datasets, same split as the decision tree
from src.preprocessing import X_train_ready, X_val_ready, X_test_ready
from src.split_data import y_train, y_val, y_test, TAG

# Converting targets to NumPy arrays
y_train = y_train.to_numpy().astype(float)
y_val = y_val.to_numpy().astype(float)
y_test = y_test.to_numpy().astype(float)

# Creating the folder for our results
OUT = Path("test_result/logistic_regression")
OUT.mkdir(parents=True, exist_ok=True)

# Setting the targets we want for the selection
TARGET_RECALL = 0.50

print()
print("Logistic regression from scratch, feature set:", TAG)


# Standardising with the training mean and standard deviation only
def standardise(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


Ztr, Zva, Zte = standardise(X_train_ready, X_val_ready, X_test_ready)

# Checkpoint
assert np.allclose(Ztr.mean(axis=0), 0)
assert not np.allclose(Zva.mean(axis=0), 0)
print("OK - standardisation uses the training mean")


# Squashing the score into a probability
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# Predicting the purchase probability
def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# Calculating the weighted cross entropy with an optional ridge penalty
def cost_logistic(X, y, w, b, sample_weight=None, lam=0.0, eps=1e-12):
    m = len(y)

    if sample_weight is None:
        sample_weight = np.ones(m)

    p = np.clip(predict_proba(X, w, b), eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    return np.sum(sample_weight * loss) / m + lam / (2 * m) * (w @ w)


# Calculating the gradient, prediction error times feature
def gradient_logistic(X, y, w, b, sample_weight=None, lam=0.0):
    m = len(y)

    if sample_weight is None:
        sample_weight = np.ones(m)

    error = (predict_proba(X, w, b) - y) * sample_weight

    dw = X.T @ error / m + lam * w / m
    db = np.sum(error) / m

    return dw, db


# Checking a gradient with finite differences
def numerical_gradient(cost, X, y, w, b, h=1e-4):
    dw = np.zeros_like(w)

    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        dw[j] = (cost(X, y, w + e, b) - cost(X, y, w - e, b)) / (2 * h)

    db = (cost(X, y, w, b + h) - cost(X, y, w, b - h)) / (2 * h)

    return dw, db


# Checkpoint, the cost of the all zero model is log 2
J0 = cost_logistic(Ztr, y_train, np.zeros(Ztr.shape[1]), 0.0)
assert np.isclose(J0, np.log(2))
print("OK - initial cost", round(J0, 4), "is log 2")

# Checkpoint, comparing our gradient with the numerical one, with weights and ridge
rng = np.random.default_rng(42)
w_test = rng.normal(size=Ztr.shape[1]) * 0.1
b_test = 0.3
sw_test = np.where(y_train == 1, 3.0, 1.0)

dw, db = gradient_logistic(Ztr, y_train, w_test, b_test, sw_test, lam=10.0)
dw_num, db_num = numerical_gradient(
    lambda X, y, w, b: cost_logistic(X, y, w, b, sw_test, lam=10.0),
    Ztr, y_train, w_test, b_test
)
rel = np.linalg.norm(np.r_[dw - dw_num, db - db_num]) / np.linalg.norm(np.r_[dw_num, db_num])
assert rel < 1e-6
print("OK - gradient, relative error", f"{rel:.1e}")


# Running gradient descent
def gradient_descent(X, y, cost, gradient, alpha=0.1, n_iters=1000):
    w, b = np.zeros(X.shape[1]), 0.0
    history = []

    for it in range(n_iters):
        dw, db = gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        history.append(cost(X, y, w, b))

    return w, b, np.array(history)


# Training one model with a safe learning rate, alpha = 1 / L
def train_model(X, y, purchase_weight, lam, n_iters=2000):
    sample_weight = np.where(y == 1, purchase_weight, 1.0)

    # Finding the largest curvature L of the cost
    L = np.linalg.eigvalsh((X * sample_weight[:, None]).T @ X / len(y)).max() / 4
    L = L + lam / len(y)

    cost = lambda X_, y_, w_, b_: cost_logistic(X_, y_, w_, b_, sample_weight, lam)
    gradient = lambda X_, y_, w_, b_: gradient_logistic(X_, y_, w_, b_, sample_weight, lam)

    return gradient_descent(X, y, cost, gradient, alpha=1 / L, n_iters=n_iters)


# Checkpoint, the cost should go down at every step
w_check, b_check, hist_check = train_model(Ztr, y_train, 3.0, 10.0, n_iters=300)
assert hist_check[-1] < hist_check[0]
assert np.all(np.diff(hist_check) <= 1e-9)
print("OK - gradient descent lowers the cost at every step")


# Calculating classification results from probabilities
def classification_metrics(y_true, p_hat, threshold=0.5):
    y_pred = (p_hat >= threshold).astype(float)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    scores = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
    counts = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    return scores, counts


# Checkpoint
scores_check, counts_check = classification_metrics(
    np.array([0, 0, 1, 1.0]), np.array([0.1, 0.9, 0.9, 0.1])
)
assert np.isclose(scores_check["Precision"], 0.5) and np.isclose(scores_check["Recall"], 0.5)
print("OK - classification metrics")


# Searching weights, ridge strength and thresholds on validation data
purchase_ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
purchase_weights = [1.0, 2.0, 3.0, 4.0, purchase_ratio]
lambdas = [0.0, 1.0, 10.0, 100.0]
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

records = []
models = {}

for purchase_weight in purchase_weights:
    for lam in lambdas:
        w, b, history = train_model(Ztr, y_train, purchase_weight, lam)
        models[(purchase_weight, lam)] = (w, b, history)

        p_val = predict_proba(Zva, w, b)

        for threshold in thresholds:
            scores, counts = classification_metrics(y_val, p_val, threshold)
            records.append({
                "Purchase_weight": purchase_weight,
                "Lambda": lam,
                "Threshold": threshold,
                **scores
            })

validation_table = pd.DataFrame(records)
validation_table.to_csv(OUT / f"validation_results_{TAG}.csv", index=False)
print("Configurations tested on validation:", len(validation_table))


# Selecting the models using validation data only
equal_row = validation_table[
    np.isclose(validation_table["Purchase_weight"], 1.0)
    & np.isclose(validation_table["Lambda"], 0.0)
    & np.isclose(validation_table["Threshold"], 0.5)
].iloc[0]

best_f1_row = validation_table.loc[validation_table["F1"].idxmax()]

enough_recall = validation_table[validation_table["Recall"] >= TARGET_RECALL]

if len(enough_recall) > 0:
    best_precision_row = enough_recall.loc[enough_recall["Precision"].idxmax()]
else:
    best_precision_row = best_f1_row

selected = pd.DataFrame([
    {"Method": "Plain logistic regression", **equal_row},
    {"Method": "Best F1", **best_f1_row},
    {"Method": "Max precision with recall of at least 0.5", **best_precision_row},
])

selected.to_csv(OUT / f"selected_models_{TAG}.csv", index=False)

print()
print("Selected models on validation:")
print(selected.round(4).to_string(index=False))


# Testing the selected models once and comparing with scikit-learn
results = []
test_predictions = {}

for _, row in selected.iterrows():
    purchase_weight = float(row["Purchase_weight"])
    lam = float(row["Lambda"])
    threshold = float(row["Threshold"])

    w, b, history = models[(purchase_weight, lam)]
    p_test = predict_proba(Zte, w, b)
    scores, counts = classification_metrics(y_test, p_test, threshold)

    # Training scikit-learn with the same weights and ridge strength
    sample_weight = np.where(y_train == 1, purchase_weight, 1.0)
    C = 1.0 / lam if lam > 0 else 1e10

    sk = LogisticRegression(C=C, max_iter=10000)
    sk.fit(Ztr, y_train, sample_weight=sample_weight)
    p_sk = sk.predict_proba(Zte)[:, 1]
    sk_scores, sk_counts = classification_metrics(y_test, p_sk, threshold)

    results.append({"Model": "Our logistic regression", "Method": row["Method"],
                    "Purchase_weight": purchase_weight, "Lambda": lam,
                    "Threshold": threshold, **scores, **counts})
    results.append({"Model": "Scikit-learn", "Method": row["Method"],
                    "Purchase_weight": purchase_weight, "Lambda": lam,
                    "Threshold": threshold, **sk_scores, **sk_counts})

    test_predictions[row["Method"]] = counts

    # Checkpoint, our probabilities should be close to scikit-learn
    if lam > 0:
        gap = np.max(np.abs(p_test - p_sk))
        assert gap < 0.05
        print("OK - probabilities close to scikit-learn for", row["Method"], "| max gap", round(gap, 4))

test_table = pd.DataFrame(results)
test_table.to_csv(OUT / f"test_results_{TAG}.csv", index=False)

print()
print("Final test results:")
print(test_table[["Model", "Method", "Purchase_weight", "Lambda", "Threshold",
                  "Accuracy", "Precision", "Recall", "F1"]].round(4).to_string(index=False))


# Comparing with the baseline that never predicts a purchase
baseline_scores, _ = classification_metrics(y_test, np.zeros(len(y_test)))

print()
print("Baseline test results:")
print({k: round(float(v), 4) for k, v in baseline_scores.items()})


# Creating graph 1, precision and recall for each threshold on validation data
chosen_w = float(best_precision_row["Purchase_weight"])
chosen_lam = float(best_precision_row["Lambda"])
w_sel, b_sel, history_sel = models[(chosen_w, chosen_lam)]
p_val_sel = predict_proba(Zva, w_sel, b_sel)

P, R = [], []
for t in thresholds:
    s, _ = classification_metrics(y_val, p_val_sel, t)
    P.append(s["Precision"])
    R.append(s["Recall"])

plt.figure(figsize=(7, 4))
plt.plot(thresholds, P, label="Precision")
plt.plot(thresholds, R, label="Recall")
plt.axhline(TARGET_RECALL, ls=":", color="gray")
plt.xlabel("Threshold")
plt.ylabel("Validation score")
plt.title("The precision and recall trade off (" + TAG + ")")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / f"threshold_trade_off_{TAG}.png", dpi=150)
plt.close()

# Creating graph 2, the learning curve of the selected model
plt.figure(figsize=(7, 4))
plt.plot(history_sel)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Training cost")
plt.title("Learning curve of the selected model (" + TAG + ")")
plt.tight_layout()
plt.savefig(OUT / f"learning_curve_{TAG}.png", dpi=150)
plt.close()

# Creating graph 3, precision and recall of every tested setting
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    validation_table["Recall"], validation_table["Precision"],
    c=validation_table["Purchase_weight"], cmap="viridis", s=8
)
plt.colorbar(scatter, label="Purchase weight")
plt.axvline(TARGET_RECALL, ls=":", color="gray")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation precision and recall (" + TAG + ")")
plt.tight_layout()
plt.savefig(OUT / f"precision_recall_{TAG}.png", dpi=150)
plt.close()

# Creating graph 4, confusion matrices of our three selected models on test data
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (method, c) in zip(axes, test_predictions.items()):
    matrix = [[c["TN"], c["FP"]], [c["FN"], c["TP"]]]
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["No purchase", "Purchase"])
    ax.set_yticks([0, 1], ["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(method, fontsize=9)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(OUT / f"confusion_matrices_{TAG}.png", dpi=150)
plt.close()

# Checkpoints
assert len(test_table) == 6
assert test_table["F1"].between(0, 1).all()
assert test_table["Accuracy"].between(0, 1).all()

print()
print("Graphs and tables saved in", OUT)
print("OK - logistic regression completed")
