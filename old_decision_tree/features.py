import pandas as pd

# Getting data from online_shoppers_intention, naming data
data = pd.read_csv("data/online_shoppers_intention.csv")
# We want to predict whether a user will make a purchase or not, so we will set the target column to "Revenue"
target = "Revenue"
# Selecting numerical variables
numbers = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]
# Selecting categorical variables
categories = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
# Combining numerical and categorical variables into a single list of selected columns
features = numbers + categories

# X contains the selected input variables
# y contains the purchase outcome, converted to 0 or 1
X = data[features]
y = data[target].astype(int)

# Displaying the total number of browsing sessions, number of input variables, and counts of numerical and categorical variables
print("Total browsing sessions:", len(data))
print("Number of input variables:", X.shape[1])
print("Numerical variables:", len(numbers))
print("Categorical variables:", len(categories))

# Show selected variables
print()
print("Selected input variables:")
for name in features:
    print(name)

# Show excluded variables
print()
print("Excluded variables:")
for name in data.columns:
    if name not in features:
        print(name)

# Show the first five shopping sessions
print()
print("First five shopping sessions:")
print(X.head())


# Show purchase distribution
print()
print("Purchases and non-purchases:")
print(y.value_counts())