import pandas as pd
import matplotlib.pyplot as plt

from src.split_data import TAG

# Getting the validation results saved by run_project
table = pd.read_csv(f"results/validation_results_{TAG}.csv")
ours = table[table["Model"] == "Our tree"]
sklearn = table[table["Model"] == "Scikit-learn"]

# Finding the best F1 for each depth
our_best = ours.groupby("Depth")["F1"].max()
sklearn_best = sklearn.groupby("Depth")["F1"].max()

# Creating the first graph
plt.figure(figsize=(7, 4))

plt.plot(our_best.index, our_best.values, "o-", label="Our decision tree")
plt.plot(sklearn_best.index, sklearn_best.values, "o-", label="Scikit-learn")

plt.xlabel("Maximum tree depth")
plt.ylabel("Best validation F1-score")
plt.title("Validation F1-score by Tree Depth (" + TAG + ")")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/model_comparison_{TAG}.png", dpi=150)
plt.close()

# Creating the second graph with precision and recall of all our settings
plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    ours["Recall"],
    ours["Precision"],
    c=ours["Purchase_weight"],
    cmap="viridis",
    s=25
)

plt.colorbar(scatter, label="Purchase weight")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision and Recall of our tree (" + TAG + ")")
plt.tight_layout()
plt.savefig(f"results/precision_recall_{TAG}.png", dpi=150)
plt.close()

# Confirming that the graphs were saved
print("Graphs saved in the results folder")
