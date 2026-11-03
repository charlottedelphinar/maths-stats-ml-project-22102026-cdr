import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# Hiding the scikit-learn warning, we stop its training at our best step on purpose
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Getting the prepared datasets, same split as the decision tree
from src.preprocessing import X_train_ready, X_val_ready, X_test_ready
from src.split_data import y_train, y_val, y_test, TAG

# Converting targets to NumPy arrays
y_train = y_train.to_numpy().astype(float)
y_val = y_val.to_numpy().astype(float)
y_test = y_test.to_numpy().astype(float)

# Creating the folder for our results
OUT = Path("test_result/neural_network")
OUT.mkdir(parents=True, exist_ok=True)

# Setting the targets we want for the selection
TARGET_RECALL = 0.50

print()
print("Neural network from scratch, feature set:", TAG)

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

# Starting the network, one hidden layer of tanh units and one sigmoid output
def init_params(n_features, n_hidden, seed=42):
    rng = np.random.default_rng(seed)

    # Using small start values because we have many input columns
    W1 = rng.standard_normal((n_features, n_hidden)) * 0.5 / np.sqrt(n_features / 2)
    b1 = np.zeros(n_hidden)
    w2 = rng.standard_normal(n_hidden) * 0.5
    b2 = 0.0

    return [W1, b1, w2, b2]

# Running the forward pass
def forward(X, params):
    W1, b1, w2, b2 = params
    A1 = np.tanh(X @ W1 + b1)
    p = sigmoid(A1 @ w2 + b2)

    return A1, p

# Predicting the purchase probability
def predict_proba(X, params):
    return forward(X, params)[1]

# Calculating the weighted cross entropy with an optional ridge penalty
def cost_nn(X, y, params, sample_weight=None, lam=0.0, eps=1e-12):
    m = len(y)

    if sample_weight is None:
        sample_weight = np.ones(m)

    p = np.clip(predict_proba(X, params), eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    W1, b1, w2, b2 = params

    return np.sum(sample_weight * loss) / m + lam / (2 * m) * (np.sum(W1 ** 2) + w2 @ w2)

# Running the backward pass, the chain rule from the loss back to the weights
def gradients_nn(X, y, params, sample_weight=None, lam=0.0):
    m = len(y)

    if sample_weight is None:
        sample_weight = np.ones(m)

    W1, b1, w2, b2 = params
    A1, p = forward(X, params)

    dz2 = (p - y) * sample_weight / m
    dw2 = A1.T @ dz2 + lam * w2 / m
    db2 = np.sum(dz2)

    dA1 = np.outer(dz2, w2)
    dZ1 = dA1 * (1 - A1 ** 2)
    dW1 = X.T @ dZ1 + lam * W1 / m
    db1 = dZ1.sum(axis=0)

    return [dW1, db1, dw2, db2]

# Checking the gradients with finite differences on a small network
def numerical_gradients(cost, params, h=1e-5):
    grads = []

    for k, param in enumerate(params):
        param = np.asarray(param, dtype=float)
        grad = np.zeros_like(param)

        for index in np.ndindex(*param.shape):
            plus = [np.array(q, dtype=float) for q in params]
            minus = [np.array(q, dtype=float) for q in params]
            plus[k][index] += h
            minus[k][index] -= h
            grad[index] = (cost(plus) - cost(minus)) / (2 * h)

        grads.append(grad)

    return grads

# Checkpoint, comparing backpropagation with the numerical gradient
rng = np.random.default_rng(0)
X_small = Ztr[:100, :6]
y_small = y_train[:100]
sw_small = np.where(y_small == 1, 3.0, 1.0)
params_small = init_params(6, 4, seed=1)
params_small[3] = 0.2

grads = gradients_nn(X_small, y_small, params_small, sw_small, lam=5.0)
grads_num = numerical_gradients(
    lambda q: cost_nn(X_small, y_small, q, sw_small, lam=5.0), params_small
)

a = np.concatenate([np.ravel(g) for g in grads])
b = np.concatenate([np.ravel(g) for g in grads_num])
rel = np.linalg.norm(a - b) / np.linalg.norm(b)
assert rel < 1e-6
print("OK - backpropagation, relative error", f"{rel:.1e}")

# Checkpoint, all zero output weights give a cost of log 2
zero_params = init_params(Ztr.shape[1], 4)
zero_params[2] = np.zeros(4)
assert np.isclose(cost_nn(Ztr, y_train, zero_params), np.log(2))
print("OK - initial cost is log 2")

# Training the network with gradient descent and keeping the best validation cost
def train_nn(X, y, Xv, yv, purchase_weight, n_hidden, lam, lr=0.1, n_iters=1500):
    sample_weight = np.where(y == 1, purchase_weight, 1.0)
    sample_weight_val = np.where(yv == 1, purchase_weight, 1.0)

    params = init_params(X.shape[1], n_hidden)
    history = []

    best_cost = np.inf
    best_params = None
    best_iter = 0

    for it in range(1, n_iters + 1):
        grads = gradients_nn(X, y, params, sample_weight, lam)

        for q, g in zip(params, grads):
            q -= lr * g

        history.append(cost_nn(X, y, params, sample_weight, lam))

        # Checking the validation cost every 25 steps, this is our early stopping
        if it % 25 == 0:
            val_cost = cost_nn(Xv, yv, params, sample_weight_val)

            if val_cost < best_cost:
                best_cost = val_cost
                best_params = [np.array(q, dtype=float) for q in params]
                best_iter = it

    return best_params, np.array(history), best_iter

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

# Checkpoint, a short training run should lower the cost
_, hist_check, _ = train_nn(Ztr, y_train, Zva, y_val, 3.0, 8, 10.0, n_iters=50)
assert hist_check[-1] < hist_check[0]
print("OK - training lowers the cost")

# Searching hidden units, weights, ridge strength and thresholds on validation data
purchase_ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
hidden_sizes = [8, 16, 32]
purchase_weights = [1.0, 3.0, purchase_ratio]
lambdas = [0.0, 10.0, 100.0]
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

records = []
models = {}

for n_hidden in hidden_sizes:
    for purchase_weight in purchase_weights:
        for lam in lambdas:
            params, history, best_iter = train_nn(
                Ztr, y_train, Zva, y_val, purchase_weight, n_hidden, lam
            )
            models[(n_hidden, purchase_weight, lam)] = (params, history, best_iter)

            p_val = predict_proba(Zva, params)

            for threshold in thresholds:
                scores, counts = classification_metrics(y_val, p_val, threshold)
                records.append({
                    "Hidden_units": n_hidden,
                    "Purchase_weight": purchase_weight,
                    "Lambda": lam,
                    "Best_iteration": best_iter,
                    "Threshold": threshold,
                    **scores
                })

validation_table = pd.DataFrame(records)
validation_table.to_csv(OUT / f"validation_results_{TAG}.csv", index=False)
print("Configurations tested on validation:", len(validation_table))

# Selecting the models using validation data only
plain = validation_table[
    (validation_table["Hidden_units"] == 16)
    & np.isclose(validation_table["Purchase_weight"], 1.0)
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
    {"Method": "Plain neural network", **plain},
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
    n_hidden = int(row["Hidden_units"])
    purchase_weight = float(row["Purchase_weight"])
    lam = float(row["Lambda"])
    threshold = float(row["Threshold"])

    params, history, best_iter = models[(n_hidden, purchase_weight, lam)]
    p_test = predict_proba(Zte, params)
    scores, counts = classification_metrics(y_test, p_test, threshold)

    # Training the scikit-learn network with the same size, weights and ridge strength
    sample_weight = np.where(y_train == 1, purchase_weight, 1.0)

    sk = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        activation="tanh",
        alpha=lam / len(y_train),
        max_iter=best_iter,
        random_state=42,
    )

    try:
        sk.fit(Ztr, y_train, sample_weight=sample_weight)
    except TypeError:
        # Older versions of scikit-learn cannot use sample weights in the network
        sk.fit(Ztr, y_train)

    p_sk = sk.predict_proba(Zte)[:, 1]
    sk_scores, sk_counts = classification_metrics(y_test, p_sk, threshold)

    results.append({"Model": "Our neural network", "Method": row["Method"],
                    "Hidden_units": n_hidden, "Purchase_weight": purchase_weight,
                    "Lambda": lam, "Threshold": threshold, **scores, **counts})
    results.append({"Model": "Scikit-learn", "Method": row["Method"],
                    "Hidden_units": n_hidden, "Purchase_weight": purchase_weight,
                    "Lambda": lam, "Threshold": threshold, **sk_scores, **sk_counts})

    test_predictions[row["Method"]] = counts

test_table = pd.DataFrame(results)
test_table.to_csv(OUT / f"test_results_{TAG}.csv", index=False)

print()
print("Final test results:")
print(test_table[["Model", "Method", "Hidden_units", "Purchase_weight", "Lambda",
                  "Threshold", "Accuracy", "Precision", "Recall", "F1"]].round(4).to_string(index=False))

# Comparing with the baseline that never predicts a purchase
baseline_scores, _ = classification_metrics(y_test, np.zeros(len(y_test)))

print()
print("Baseline test results:")
print({k: round(float(v), 4) for k, v in baseline_scores.items()})

# Creating graph 1, precision and recall for each threshold on validation data
chosen = (int(best_precision_row["Hidden_units"]),
          float(best_precision_row["Purchase_weight"]),
          float(best_precision_row["Lambda"]))
params_sel, history_sel, best_iter_sel = models[chosen]
p_val_sel = predict_proba(Zva, params_sel)

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

# Creating graph 2, the learning curve of the selected network
plt.figure(figsize=(7, 4))
plt.plot(history_sel)
plt.axvline(best_iter_sel, ls=":", color="gray", label="Best validation step")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Training cost")
plt.title("Learning curve of the selected network (" + TAG + ")")
plt.legend()
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
print("OK - neural network completed")
