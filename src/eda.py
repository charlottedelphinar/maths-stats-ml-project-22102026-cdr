print("Exploratory Data Analysis (EDA) for Online Shoppers Purchasing Intention Dataset")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/online_shoppers_intention.csv")  
#1 Dimensions of the dataset
print(df.shape)
#2 First few rows
print(df.head())
#3 Data types
print(df.dtypes)
#4 Missing values
print(df.isna().sum())
#5 Duplicated rows
print(df.duplicated().sum())
#6 Purchases and non-purchases
print(df["Revenue"].value_counts())
#7 % of sessions that resulted in a purchase
print(round(df["Revenue"].mean() * 100, 2))
#Descriptive statistics
print(df.describe())
#Bar chart of the target distribution
counts = df["Revenue"].value_counts().reindex(
    [False, True], fill_value=0
)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["No purchase", "Purchase"], counts.values)
# Graph saved in results/folder w title and close figure descriptive title, a y-axis label and the exact counts displayed above each bar.
# Create and save the target distribution chart
from pathlib import Path

Path("results").mkdir(exist_ok=True)

counts = df["Revenue"].value_counts().reindex(
    [False, True], fill_value=0
)

fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.bar(
    ["No purchase", "Purchase"],
    counts.values
)

ax.set_title("Distribution of Purchase Outcomes")
ax.set_xlabel("Session outcome")
ax.set_ylabel("Number of sessions")

ax.bar_label(bars, padding=4)
ax.set_ylim(0, counts.max() * 1.15)

fig.tight_layout()
fig.savefig("results/target_distribution.png", dpi=160)
plt.close(fig)

print("Graph saved: results/target_distribution.png")
