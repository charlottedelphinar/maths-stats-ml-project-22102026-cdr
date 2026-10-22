
import numpy as np
import pandas as pd

# Get the datasets from part 2
from src.split_data import X_train, X_val, X_test

# Work with copies of the datasets
train = X_train.copy()
val = X_val.copy()
test = X_test.copy()

# Define categories as variables
categories = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Convert categories to text
for column in categories:
    train[column] = train[column].astype(str)
    val[column] = val[column].astype(str)
    test[column] = test[column].astype(str)

# Convert categories into numerical columns
train_encoded = pd.get_dummies(train, columns=categories, dtype=float)
val_encoded = pd.get_dummies(val, columns=categories, dtype=float)
test_encoded = pd.get_dummies(test, columns=categories, dtype=float)

# Use the same columns as the training dataset
val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)

# Convert the datasets into NumPy arrays
X_train_ready = train_encoded.to_numpy(dtype=float)
X_val_ready = val_encoded.to_numpy(dtype=float)
X_test_ready = test_encoded.to_numpy(dtype=float)

# Check the number of rows and columns
print()
print("Training shape:", X_train_ready.shape)
print("Validation shape:", X_val_ready.shape)
print("Test shape:", X_test_ready.shape)

# Check that all datasets have the same number of columns
assert X_train_ready.shape[1] == X_val_ready.shape[1]
assert X_train_ready.shape[1] == X_test_ready.shape[1]

# Check that all values are valid numbers
assert np.isfinite(X_train_ready).all()
assert np.isfinite(X_val_ready).all()
assert np.isfinite(X_test_ready).all()

# Check that no shopping sessions were lost
assert X_train_ready.shape[0] == len(X_train)
assert X_val_ready.shape[0] == len(X_val)
assert X_test_ready.shape[0] == len(X_test)

print("OK - preprocessing completed")
