import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.multitest import multipletests

files = {
    "Median": "results/median_test_predictions.csv",
    "LightGBM": "results/lightgbm_test_predictions.csv",
    "Polynomial Regression": "results/tables/polynomial_regression_test_predictions.csv",
    "Time Features": "results/time_features_test_predictions.csv",
    "XGBoost": "results/xgboost_test_predictions.csv",
}

frames = {}

for name, path in files.items():
    df = pd.read_csv(path).sort_values("hour").reset_index(drop=True)
    frames[name] = df

# Confirm that all models use identical test timestamps and actual values.
reference = frames["Median"]
for name, df in frames.items():
    if not reference["hour"].equals(df["hour"]):
        raise ValueError(f"{name}: test timestamps do not match.")
    if not np.allclose(reference["actual_cooling_kw"], df["actual_cooling_kw"]):
        raise ValueError(f"{name}: actual test values do not match.")

actual = reference["actual_cooling_kw"].to_numpy()

print("MODEL METRICS")
for name, df in frames.items():
    pred = df["predicted_cooling_kw"].to_numpy()
    print(
        f"{name:22s} "
        f"MAE={mean_absolute_error(actual, pred):.4f} kW | "
        f"RMSE={np.sqrt(mean_squared_error(actual, pred)):.4f} kW | "
        f"R²={r2_score(actual, pred):.4f}"
    )

print("\nPAIRED WILCOXON TESTS: absolute errors vs Median")
median_errors = np.abs(
    actual - frames["Median"]["predicted_cooling_kw"].to_numpy()
)

comparisons = []

for name, df in frames.items():
    if name == "Median":
        continue

    errors = np.abs(actual - df["predicted_cooling_kw"].to_numpy())
    stat, p_value = wilcoxon(errors, median_errors)

    # Positive means this model has lower average absolute error than Median.
    median_error_difference = np.median(median_errors - errors)

    comparisons.append({
        "Model": name,
        "Median_paired_error_difference_kW": median_error_difference,
        "p_value": p_value,
    })

p_values = [row["p_value"] for row in comparisons]
_, adjusted_p_values, _, _ = multipletests(p_values, method="holm")

for row, adjusted_p in zip(comparisons, adjusted_p_values):
    row["Holm_adjusted_p_value"] = adjusted_p

results = pd.DataFrame(comparisons)
print(results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

results.to_csv("results/tables/statistical_comparison.csv", index=False)
print("\nSaved: results/tables/statistical_comparison.csv")