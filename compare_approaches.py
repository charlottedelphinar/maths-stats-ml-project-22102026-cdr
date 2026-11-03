import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

# Creating the folder for the comparison
OUT = Path("test_result/comparison")
OUT.mkdir(parents=True, exist_ok=True)

# Giving the same name to the three selected models of every approach
def common_name(method):
    if method.startswith("Equal") or method.startswith("Plain"):
        return "1 Plain model"
    if method.startswith("Best F1"):
        return "2 Best F1"
    return "3 Max precision, recall of at least 0.5"

# Collecting the test results of the three approaches for each feature set
for tag in ["strict", "pagevalues"]:
    files = {
        "Decision tree": Path(f"results/test_results_{tag}.csv"),
        "Logistic regression": Path(f"test_result/logistic_regression/test_results_{tag}.csv"),
        "Neural network": Path(f"test_result/neural_network/test_results_{tag}.csv"),
    }

    rows = []

    for approach, path in files.items():
        if not path.exists():
            print("Missing file, skipping:", path)
            continue

        table = pd.read_csv(path)

        # Keeping only our from scratch models, not the scikit-learn benchmark
        ours = table[table["Model"].str.startswith("Our")].copy()
        ours["Approach"] = approach
        ours["Selected model"] = ours["Method"].apply(common_name)
        rows.append(ours[["Approach", "Selected model", "Accuracy", "Precision", "Recall", "F1",
                          "TN", "FP", "FN", "TP"]])

    if not rows:
        continue

    comparison = pd.concat(rows, ignore_index=True).sort_values(["Selected model", "Approach"])
    comparison[["TN", "FP", "FN", "TP"]] = comparison[["TN", "FP", "FN", "TP"]].astype(int)
    comparison.to_csv(OUT / f"approach_comparison_{tag}.csv", index=False)

    print()
    print("Test results of the three approaches, feature set:", tag)
    print(comparison.round(4).to_string(index=False))

    # Creating the graph, one group of bars for each selected model
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, name in zip(axes, sorted(comparison["Selected model"].unique())):
        part = comparison[comparison["Selected model"] == name].set_index("Approach")
        part[["Precision", "Recall", "F1"]].plot.bar(ax=ax, rot=20, legend=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Test score")
    axes[0].legend(["Precision", "Recall", "F1"])
    fig.suptitle("Test results of the three approaches (" + tag + ")")
    fig.tight_layout()
    fig.savefig(OUT / f"approach_comparison_{tag}.png", dpi=150)
    plt.close(fig)

    # Creating the confusion matrices, one row for each selected model and one column for each approach
    models = sorted(comparison["Selected model"].unique())
    approaches = list(comparison["Approach"].unique())

    fig, axes = plt.subplots(len(models), len(approaches), figsize=(5 * len(approaches), 4.3 * len(models)))

    for i, model in enumerate(models):
        for j, approach in enumerate(approaches):
            row = comparison[(comparison["Selected model"] == model) & (comparison["Approach"] == approach)].iloc[0]
            matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)

            ConfusionMatrixDisplay(matrix, display_labels=["No purchase", "Purchase"]).plot(
                ax=axes[i, j], colorbar=False
            )
            axes[i, j].set_title(approach + "\n" + model[2:], fontsize=10)

    fig.suptitle("Confusion matrices on the test set (" + tag + ")", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_matrices_all_{tag}.png", dpi=150)
    plt.close(fig)

    # Creating one confusion matrix for the best result, the highest test F1
    best = comparison.loc[comparison["F1"].idxmax()]
    matrix = np.array([[best["TN"], best["FP"]], [best["FN"], best["TP"]]], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(matrix, display_labels=["No purchase", "Purchase"]).plot(ax=ax, colorbar=False)
    ax.set_title(
        "Best result (" + tag + "): " + best["Approach"] + ", " + best["Selected model"][2:] + "\n"
        + f"Precision {best['Precision']:.3f}   Recall {best['Recall']:.3f}   F1 {best['F1']:.3f}",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(OUT / f"best_result_{tag}.png", dpi=150)
    plt.close(fig)

    print()
    print("Best result for", tag, ":", best["Approach"], "|", best["Selected model"][2:], "| F1", round(best["F1"], 4))

print()
print("OK - comparison saved in", OUT)
