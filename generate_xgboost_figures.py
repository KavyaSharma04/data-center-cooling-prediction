
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load XGBoost predictions
df = pd.read_csv("results/xgboost_test_predictions.csv")

df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour")

actual = df["actual_cooling_kw"]
predicted = df["predicted_cooling_kw"]

# Prediction error: actual - predicted
df["error"] = actual - predicted

# Create output directory
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------
# Figure 1: Actual vs Predicted
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.45,
    s=15
)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("Actual Cooling Power (kW)")
plt.ylabel("Predicted Cooling Power (kW)")
plt.title("XGBoost: Actual vs Predicted Cooling Power")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{output_dir}/xgboost_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------
# Figure 2: Prediction Errors Over Time
# --------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["hour"],
    df["error"],
    linewidth=0.7
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Time")
plt.ylabel("Prediction Error (kW)")
plt.title("XGBoost: Prediction Errors Over Time")
plt.xticks(rotation=30)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{output_dir}/xgboost_prediction_errors_over_time.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------
# Figure 3: Prediction Error Distribution
# --------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    df["error"],
    bins=50,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Prediction Error (kW)")
plt.ylabel("Frequency")
plt.title("XGBoost: Distribution of Prediction Errors")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{output_dir}/xgboost_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("All three XGBoost figures generated successfully!")

print("Saved in:", output_dir)
print("Number of test observations:", len(df))
print("Mean absolute error:", df["error"].abs().mean())
print("Root mean squared error:", np.sqrt((df["error"] ** 2).mean()))
