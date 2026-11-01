import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load saved predictions from the median model
df = pd.read_csv("results/median_test_predictions.csv")

df["hour"] = pd.to_datetime(df["hour"], utc=True)

actual = df["actual_cooling_kw"]
predicted = df["predicted_cooling_kw"]

# Calculate prediction errors
df["error"] = actual - predicted
df["absolute_error"] = df["error"].abs()

print("========== OVERALL TEST PERFORMANCE ==========")
print("MAE:", round(mean_absolute_error(actual, predicted), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 4))
print("R²:", round(r2_score(actual, predicted), 4))

print("\n========== ERROR SUMMARY ==========")
print(df["error"].describe())

print("\n========== ERROR BY ACTUAL COOLING POWER ==========")

df["cooling_range"] = pd.cut(
    actual,
    bins=[-np.inf, 5, 10, 20, 30, np.inf],
    labels=["Below 5", "5–10", "10–20", "20–30", "Above 30"]
)

summary = df.groupby(
    "cooling_range", observed=False
).agg(
    Records=("actual_cooling_kw", "count"),
    Mean_Actual=("actual_cooling_kw", "mean"),
    Mean_Predicted=("predicted_cooling_kw", "mean"),
    MAE=("absolute_error", "mean")
)

print(summary.round(3))

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.25)

max_value = max(actual.max(), predicted.max())
plt.plot([0, max_value], [0, max_value], linestyle="--")

plt.xlabel("Actual Cooling Power (kW)")
plt.ylabel("Predicted Cooling Power (kW)")
plt.title("Actual vs Predicted Cooling Power")
plt.tight_layout()

plt.savefig("results/actual_vs_predicted_median.png", dpi=300)
plt.show()

# Plot prediction errors over time
plt.figure(figsize=(12, 5))
plt.plot(df["hour"], df["error"], linewidth=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Time")
plt.ylabel("Actual - Predicted (kW)")
plt.title("Prediction Errors Over Time")
plt.tight_layout()

plt.savefig("results/prediction_errors_over_time.png", dpi=300)
plt.show()

print("\nGraphs saved in the results folder.")