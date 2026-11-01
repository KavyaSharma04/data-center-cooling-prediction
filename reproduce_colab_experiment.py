
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import joblib


# ==========================================
# 1. LOAD PREPARED DATASET
# ==========================================

print("Loading prepared dataset...")

df = pd.read_csv(
    "results/data/hourly_merged_data.csv",
    parse_dates=["hour"]
)

df = df.sort_values("hour").reset_index(drop=True)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "it_power_kw",
    "outdoor_air_temp",
    "outdoor_air_humidity"
]

target = "cooling_kw"

X = df[features]
y = df[target]

print("\nFeatures:", features)
print("Target:", target)


# ==========================================
# 3. CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print(
    "Training period:",
    df["hour"].iloc[0],
    "to",
    df["hour"].iloc[train_end - 1]
)

print(
    "Validation period:",
    df["hour"].iloc[train_end],
    "to",
    df["hour"].iloc[val_end - 1]
)

print(
    "Testing period:",
    df["hour"].iloc[val_end],
    "to",
    df["hour"].iloc[-1]
)


# ==========================================
# 4. TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# 5. EVALUATE VALIDATION SET
# ==========================================

val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
val_r2 = r2_score(y_val, val_predictions)

print("\n========== VALIDATION RESULTS ==========")
print(f"MAE:  {val_mae:.4f} kW")
print(f"RMSE: {val_rmse:.4f} kW")
print(f"R²:   {val_r2:.4f}")


# ==========================================
# 6. EVALUATE TEST SET
# ==========================================

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\n========== TEST RESULTS ==========")
print(f"MAE:  {test_mae:.4f} kW")
print(f"RMSE: {test_rmse:.4f} kW")
print(f"R²:   {test_r2:.4f}")


# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ==========================================
# 8. SAVE MODEL AND PREDICTIONS
# ==========================================

os.makedirs("results", exist_ok=True)

joblib.dump(model, "results/random_forest_model.pkl")

test_results = pd.DataFrame({
    "hour": df["hour"].iloc[val_end:].values,
    "actual_cooling_kw": y_test.values,
    "predicted_cooling_kw": test_predictions
})

test_results.to_csv(
    "results/test_predictions.csv",
    index=False
)

print("\nModel saved to results/random_forest_model.pkl")
print("Predictions saved to results/test_predictions.csv")

print("\nExperiment completed successfully!")


print("\n========== COOLING POWER BY TIME PERIOD ==========")

df["period"] = pd.cut(
    df["hour"],
    bins=[
        pd.Timestamp("2016-01-01", tz="UTC"),
        pd.Timestamp("2023-01-05", tz="UTC"),
        pd.Timestamp("2024-05-14", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC")
    ],
    labels=["Training period", "Validation period", "Test period"]
)

period_summary = df.groupby(
    "period",
    observed=False
)["cooling_kw"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(period_summary.round(3))


print("\n========== EXTREME VALUE INVESTIGATION ==========")

thresholds = [30, 40, 50, 75, 100]

for threshold in thresholds:
    count = (df["cooling_kw"] > threshold).sum()
    print(f"Cooling power > {threshold} kW: {count} records")

print("\nTop 20 cooling-power observations:")

extreme_records = df.nlargest(20, "cooling_kw")[
    [
        "hour",
        "cooling_kw",
        "it_power_kw",
        "outdoor_air_temp",
        "outdoor_air_humidity"
    ]
]

print(extreme_records.to_string(index=False))

print("\n========== EXTREME VALUES BY PERIOD ==========")

for period in ["Training period", "Validation period", "Test period"]:
    period_df = df[df["period"] == period]

    print(f"\n{period}")
    print("Records above 30 kW:", (period_df["cooling_kw"] > 30).sum())
    print("Records above 40 kW:", (period_df["cooling_kw"] > 40).sum())
    print("Records above 50 kW:", (period_df["cooling_kw"] > 50).sum())

print("\n========== INSPECTING EXTREME OBSERVATIONS ==========")

extreme_dates = [
    "2019-12-12 17:00:00+00:00",
    "2019-12-12 20:00:00+00:00"
]

for date in extreme_dates:
    timestamp = pd.Timestamp(date)

    start = timestamp - pd.Timedelta(hours=3)
    end = timestamp + pd.Timedelta(hours=3)

    nearby = df[
        (df["hour"] >= start) &
        (df["hour"] <= end)
    ][
        [
            "hour",
            "cooling_kw",
            "it_power_kw",
            "outdoor_air_temp",
            "outdoor_air_humidity"
        ]
    ]

    print(f"\nReadings around {date}:")
    print(nearby.to_string(index=False))

print("\n========== RAW POWER DATA INVESTIGATION ==========")

power_raw = pd.read_csv(
    "data/raw/power_file.csv",
    low_memory=False
)

cleaned_ts = power_raw["ts"].str.replace(
    r"\s*\([^)]*\)\s*$",
    "",
    regex=True
)

power_raw["ts"] = pd.to_datetime(
    cleaned_ts,
    format="%a %b %d %Y %H:%M:%S GMT%z",
    errors="coerce",
    utc=True
)

for date in [
    "2019-12-12 17:00:00+00:00",
    "2019-12-12 20:00:00+00:00"
]:
    timestamp = pd.Timestamp(date)

    start = timestamp
    end = timestamp + pd.Timedelta(hours=1)

    records = power_raw[
        (power_raw["ts"] >= start) &
        (power_raw["ts"] < end)
    ][["ts", "cooling_kw", "it_power_kw"]]

    print(f"\nRaw readings during {date}:")
    print("Number of records:", len(records))
    print(records.to_string(index=False))

    print("\nCooling-power statistics:")
    print(records["cooling_kw"].describe())