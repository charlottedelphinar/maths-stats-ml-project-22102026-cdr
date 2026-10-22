import numpy as np
import pandas as pd

# Getting data from online_shoppers_intention, naming data
data = pd.read_csv("data/online_shoppers_intention.csv")

# Our input variables
features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Separate inputs and target
X = data[features]
y = data["Revenue"].astype(int)

# Shuffle the row indices
n = len(data)
indices = np.random.default_rng(42).permutation(n)

# Calculate the number of rows
n_test = round(0.2 * n)
n_val = round(0.2 * n)

# Divide the indices into three groups
test_idx = indices[:n_test]
val_idx = indices[n_test:n_test + n_val]
train_idx = indices[n_test + n_val:]

# Select the corresponding rows
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]


# Check the sizes of the three datasets
assert len(X_train) == 7398
assert len(X_val) == 2466
assert len(X_test) == 2466

# Check that every row is used exactly once
assert len(train_idx) + len(val_idx) + len(test_idx) == n
assert len(np.unique(indices)) == n

print("OK - dataset split completed")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print()
print("Purchase rates:")
print("Training:", round(y_train.mean() * 100, 2), "%")
print("Validation:", round(y_val.mean() * 100, 2), "%")
print("Test:", round(y_test.mean() * 100, 2), "%")
