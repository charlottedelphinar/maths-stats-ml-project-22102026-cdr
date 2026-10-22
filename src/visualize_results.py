import numpy as np
import matplotlib.pyplot as plt

# Getting the results from Part 6
depths = [3, 5, 7]

our_tree = [0.0000, 0.1683, 0.1732]
sklearn_tree = [0.0000, 0.0934, 0.1102]

# Creating the graph
plt.figure(figsize=(7, 4))

# Plotting the results of both decision trees
plt.plot(depths, our_tree, "o-", label="Our decision tree")
plt.plot(depths, sklearn_tree, "o-", label="Scikit-learn")

# Adding labels and a title to the graph
plt.xlabel("Maximum tree depth")
plt.ylabel("F1-score")
plt.title("Validation F1-score by Tree Depth")

# Setting the values on the axes
plt.xticks(depths)
plt.ylim(0, 0.22)

# Adding a legend to identify the models
plt.legend()

# Adjusting the graph layout
plt.tight_layout()

# Saving the graph in the results folder
plt.savefig("results/model_comparison.png", dpi=150)

# Displaying the graph
plt.show()

# Confirming that the graph was saved
print("Graph saved: results/model_comparison.png")