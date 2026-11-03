import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Getting our decision tree functions
from src.tree_scratch import build_tree, predict

# Getting the prepared datasets, same split as everywhere else
from src.preprocessing import X_train_ready, X_val_ready, X_test_ready
from src.split_data import y_train, y_val, y_test, TAG

# Converting targets to NumPy arrays
y_tr_int = y_train.to_numpy()
y_tr = y_train.to_numpy().astype(float)
y_va = y_val.to_numpy().astype(float)
y_te = y_test.to_numpy().astype(float)

# Creating the folder for the comparison
OUT = Path("test_result/comparison")
OUT.mkdir(parents=True, exist_ok=True)

print()
print("Training versus test results, feature set:", TAG)

# Calculating classification metrics from predicted classes
def metrics(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    accuracy = np.mean(y == y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

# Giving the same name to the three selected models of every approach
def common_name(method):
    if method.startswith("Equal") or method.startswith("Plain"):
        return "1 Plain model"
    if method.startswith("Best F1"):
        return "2 Best F1"
    return "3 Max precision, recall of at least 0.5"

# Standardising with the training mean and standard deviation only
mean = X_train_ready.mean(axis=0)
std = np.where(X_train_ready.std(axis=0) == 0, 1.0, X_train_ready.std(axis=0))
Ztr, Zva, Zte = [(X - mean) / std for X in (X_train_ready, X_val_ready, X_test_ready)]

# Squashing the score into a probability
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Training logistic regression with gradient descent, same code as logreg_scratch.py
def train_logistic(X, y, purchase_weight, lam, n_iters=2000):
    m = len(y)
    sample_weight = np.where(y == 1, purchase_weight, 1.0)

    L = np.linalg.eigvalsh((X * sample_weight[:, None]).T @ X / m).max() / 4 + lam / m
    alpha = 1 / L

    w, b = np.zeros(X.shape[1]), 0.0

    for it in range(n_iters):
        error = (sigmoid(X @ w + b) - y) * sample_weight
        w = w - alpha * (X.T @ error / m + lam * w / m)
        b = b - alpha * np.sum(error) / m

    return w, b

# Starting the neural network, same code as nn_scratch.py
def init_params(n_features, n_hidden, seed=42):
    rng = np.random.default_rng(seed)

    W1 = rng.standard_normal((n_features, n_hidden)) * 0.5 / np.sqrt(n_features / 2)
    b1 = np.zeros(n_hidden)
    w2 = rng.standard_normal(n_hidden) * 0.5
    b2 = 0.0

    return [W1, b1, w2, b2]

# Running the forward pass of the neural network
def nn_proba(X, params):
    W1, b1, w2, b2 = params
    return sigmoid(np.tanh(X @ W1 + b1) @ w2 + b2)

# Calculating the weighted cross entropy of the neural network
def nn_cost(X, y, params, sample_weight, lam=0.0, eps=1e-12):
    p = np.clip(nn_proba(X, params), eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    W1, b1, w2, b2 = params

    return np.sum(sample_weight * loss) / len(y) + lam / (2 * len(y)) * (np.sum(W1 ** 2) + w2 @ w2)

# Training the neural network and keeping the best validation step
def train_nn(X, y, Xv, yv, purchase_weight, n_hidden, lam, lr=0.1, n_iters=1500):
    m = len(y)
    sample_weight = np.where(y == 1, purchase_weight, 1.0)
    sample_weight_val = np.where(yv == 1, purchase_weight, 1.0)

    params = init_params(X.shape[1], n_hidden)
    best_cost, best_params = np.inf, None

    for it in range(1, n_iters + 1):
        W1, b1, w2, b2 = params
        A1 = np.tanh(X @ W1 + b1)
        p = sigmoid(A1 @ w2 + b2)

        dz2 = (p - y) * sample_weight / m
        dw2 = A1.T @ dz2 + lam * w2 / m
        db2 = np.sum(dz2)
        dZ1 = np.outer(dz2, w2) * (1 - A1 ** 2)
        dW1 = X.T @ dZ1 + lam * W1 / m
        db1 = dZ1.sum(axis=0)

        for q, g in zip(params, [dW1, db1, dw2, db2]):
            q -= lr * g

        if it % 25 == 0:
            val_cost = nn_cost(Xv, yv, params, sample_weight_val)

            if val_cost < best_cost:
                best_cost = val_cost
                best_params = [np.array(q, dtype=float) for q in params]

    return best_params

# Getting the selected models of every approach from the saved files
files = {
    "Decision tree": Path(f"results/selected_models_{TAG}.csv"),
    "Logistic regression": Path(f"test_result/logistic_regression/selected_models_{TAG}.csv"),
    "Neural network": Path(f"test_result/neural_network/selected_models_{TAG}.csv"),
}

rows = []

for approach, path in files.items():
    selected = pd.read_csv(path)

    for _, row in selected.iterrows():
        name = common_name(row["Method"])
        purchase_weight = float(row["Purchase_weight"])

        # Predicting the three datasets with the selected model
        if approach == "Decision tree":
            tree = build_tree(
                X_train_ready, y_tr_int,
                max_depth=int(row["Depth"]),
                min_samples_leaf=int(row["Min_leaf"]),
                sample_weight=np.where(y_tr_int == 1, purchase_weight, 1.0)
            )
            predictions = [predict(X, tree) for X in (X_train_ready, X_val_ready, X_test_ready)]

        elif approach == "Logistic regression":
            w, b = train_logistic(Ztr, y_tr, purchase_weight, float(row["Lambda"]))
            predictions = [(sigmoid(Z @ w + b) >= row["Threshold"]).astype(float) for Z in (Ztr, Zva, Zte)]

        else:
            params = train_nn(Ztr, y_tr, Zva, y_va, purchase_weight,
                              int(row["Hidden_units"]), float(row["Lambda"]))
            predictions = [(nn_proba(Z, params) >= row["Threshold"]).astype(float) for Z in (Ztr, Zva, Zte)]

        for dataset, y_true, y_pred in zip(["Train", "Validation", "Test"],
                                           [y_tr, y_va, y_te], predictions):
            rows.append({"Approach": approach, "Selected model": name,
                         "Set": dataset, **metrics(y_true, y_pred)})

results = pd.DataFrame(rows)

# Checkpoint, our test numbers must match the numbers saved by the other scripts
saved_files = {
    "Decision tree": Path(f"results/test_results_{TAG}.csv"),
    "Logistic regression": Path(f"test_result/logistic_regression/test_results_{TAG}.csv"),
    "Neural network": Path(f"test_result/neural_network/test_results_{TAG}.csv"),
}

for approach, path in saved_files.items():
    saved = pd.read_csv(path)
    saved = saved[saved["Model"].str.startswith("Our")]

    for _, row in saved.iterrows():
        mine = results[(results["Approach"] == approach)
                       & (results["Selected model"] == common_name(row["Method"]))
                       & (results["Set"] == "Test")].iloc[0]
        assert np.isclose(mine["F1"], row["F1"], atol=1e-4), (approach, row["Method"])

print("OK - the test numbers match the saved results")

# Calculating the gap between training and test
wide = results.pivot_table(index=["Approach", "Selected model"], columns="Set", values="F1")
wide["Gap train minus test"] = wide["Train"] - wide["Test"]
wide["Gap validation minus test"] = wide["Validation"] - wide["Test"]

results.to_csv(OUT / f"train_validation_test_{TAG}.csv", index=False)
wide.round(4).to_csv(OUT / f"f1_gaps_{TAG}.csv")

print()
print("F1 on training, validation and test:")
print(wide.round(4).to_string())

# Calculating simple reference models on the test set
purchase_rate_test = y_te.mean()
always_purchase = metrics(y_te, np.ones(len(y_te)))
print()
print("Purchase rate in the test set:", round(purchase_rate_test, 4))
print("Always predict purchase on test:", {k: round(float(v), 4) for k, v in always_purchase.items()})

# Creating graph 1, precision, recall and F1 for training, validation and test
colors = {"Train": "#9ecae1", "Validation": "#4292c6", "Test": "#08306b"}
models = sorted(results["Selected model"].unique())
approaches = list(files.keys())
metric_names = ["Precision", "Recall", "F1"]

fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)

for r, metric in enumerate(metric_names):
    for c, model in enumerate(models):
        ax = axes[r, c]
        part = results[results["Selected model"] == model].pivot(
            index="Approach", columns="Set", values=metric
        ).loc[approaches, ["Train", "Validation", "Test"]]

        part.plot.bar(ax=ax, color=[colors[s] for s in part.columns], legend=False, rot=15, width=0.8)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=1)

        # Drawing the reference of always predicting a purchase on the test set
        if metric == "Precision":
            ax.axhline(purchase_rate_test, ls="--", color="gray", lw=1)
        if metric == "F1":
            ax.axhline(always_purchase["F1"], ls="--", color="gray", lw=1)

        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_title(metric + ", " + model[2:], fontsize=10)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=colors[s], label=s) for s in ["Train", "Validation", "Test"]]
handles.append(Line2D([0], [0], ls="--", color="gray", label="Always predict purchase (test)"))
axes[0, 0].legend(handles=handles, fontsize=8, loc="upper left")
fig.suptitle("Training, validation and test results (" + TAG + "), dashed line = always predicting a purchase",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / f"train_vs_test_{TAG}.png", dpi=150)
plt.close(fig)

# Creating graph 2, the distribution of the three datasets
sizes = [len(y_tr), len(y_va), len(y_te)]
rates = [y_tr.mean() * 100, y_va.mean() * 100, y_te.mean() * 100]
labels = ["Train", "Validation", "Test"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

bars = axes[0].bar(labels, sizes, color=[colors[s] for s in labels])
axes[0].bar_label(bars, padding=3)
axes[0].set_title("Number of sessions")
axes[0].set_ylim(0, max(sizes) * 1.15)

bars = axes[1].bar(labels, rates, color=[colors[s] for s in labels])
axes[1].bar_label(bars, fmt="%.2f %%", padding=3)
axes[1].set_title("Share of purchases")
axes[1].set_ylim(0, 25)

fig.suptitle("Distribution of the three datasets (" + TAG + ")")
fig.tight_layout()
fig.savefig(OUT / f"split_distribution_{TAG}.png", dpi=150)
plt.close(fig)

# Checkpoints
assert len(results) == 27
assert results["F1"].between(0, 1).all()

print()
print("OK - training versus test comparison completed")
