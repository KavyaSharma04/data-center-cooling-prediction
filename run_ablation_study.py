import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the same dataset used by the current XGBoost experiment
df = pd.read_csv("results/data/hourly_median_merged_data.csv")
df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

# Create time features
df["hour_of_day"] = df["hour"].dt.hour
df["day_of_week"] = df["hour"].dt.day_of_week
df["month"] = df["hour"].dt.month

target = "cooling_kw"

# Compare the full feature set with feature groups removed
experiments = {
    "Full model": [
        "it_power_kw",
        "outdoor_air_temp",
        "outdoor_air_humidity",
        "hour_of_day",
        "day_of_week",
        "month",
    ],
    "No time features": [
        "it_power_kw",
        "outdoor_air_temp",
        "outdoor_air_humidity",
    ],
    "No weather features": [
        "it_power_kw",
        "hour_of_day",
        "day_of_week",
        "month",
    ],
    "No IT power": [
        "outdoor_air_temp",
        "outdoor_air_humidity",
        "hour_of_day",
        "day_of_week",
        "month",
    ],
}

# Use the same chronological 70% / 15% / 15% split for every experiment
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

y_train = df[target].iloc[:train_end]
y_val = df[target].iloc[train_end:val_end]
y_test = df[target].iloc[val_end:]

results = []

for experiment_name, features in experiments.items():
    print(f"\nRunning: {experiment_name}")

    X_train = df[features].iloc[:train_end]
    X_val = df[features].iloc[train_end:val_end]
    X_test = df[features].iloc[val_end:]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=30,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Experiment": experiment_name,
        "Features": ", ".join(features),
        "Best_iteration": model.best_iteration,
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "R2": r2,
    })

    print(f"Test MAE:  {mae:.4f} kW")
    print(f"Test RMSE: {rmse:.4f} kW")
    print(f"Test R²:   {r2:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/tables/xgboost_ablation_results.csv", index=False)

print("\n========== ABLATION STUDY SUMMARY ==========")
print(results_df[["Experiment", "Best_iteration", "MAE_kW", "RMSE_kW", "R2"]].round(4))
print("\nSaved: results/tables/xgboost_ablation_results.csv")