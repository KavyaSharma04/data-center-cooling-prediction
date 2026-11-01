import pandas as pd
import matplotlib.pyplot as plt

# Load the median-based dataset
df = pd.read_csv("results/data/hourly_median_merged_data.csv")

df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

# Reproduce the same chronological split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df["period"] = "Test"
df.loc[:train_end - 1, "period"] = "Training"
df.loc[train_end:val_end - 1, "period"] = "Validation"

# Print summary statistics
print("========== COOLING POWER BY PERIOD ==========")

for period in ["Training", "Validation", "Test"]:
    values = df.loc[df["period"] == period, "cooling_kw"]

    print(f"\n{period}")
    print("Count:", len(values))
    print("Mean:", round(values.mean(), 3))
    print("Median:", round(values.median(), 3))
    print("Standard deviation:", round(values.std(), 3))
    print("Minimum:", round(values.min(), 3))
    print("Maximum:", round(values.max(), 3))

# Plot boxplots
plt.figure(figsize=(9, 6))

data = [
    df.loc[df["period"] == period, "cooling_kw"]
    for period in ["Training", "Validation", "Test"]
]

plt.boxplot(
    data,
    tick_labels=["Training", "Validation", "Test"],
    showfliers=False
)

plt.title("Cooling Power Distribution Across Time Periods")
plt.xlabel("Dataset Period")
plt.ylabel("Cooling Power (kW)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("results/cooling_power_distribution.png", dpi=300)
plt.show()

print("\nGraph saved to results/cooling_power_distribution.png")