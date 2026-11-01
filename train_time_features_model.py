import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the median-based dataset
print("Loading median dataset...")

df = pd.read_csv("results/data/hourly_median_merged_data.csv")

df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

# 2. Add time-based features
df["hour_of_day"] = df["hour"].dt.hour
df["day_of_week"] = df["hour"].dt.dayofweek
df["month"] = df["hour"].dt.month

features = [
    "it_power_kw",
    "outdoor_air_temp",
    "outdoor_air_humidity",
    "hour_of_day",
    "day_of_week",
    "month"
]

target = "cooling_kw"

X = df[features]
y = df[target]

# 3. Use the same chronological split
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

# 4. Train the model
print("\nTraining Random Forest with time-based features...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluate validation and test performance
def evaluate_model(name, X_part, y_part):
    predictions = model.predict(X_part)

    mae = mean_absolute_error(y_part, predictions)
    rmse = np.sqrt(mean_squared_error(y_part, predictions))
    r2 = r2_score(y_part, predictions)

    print(f"\n========== {name} RESULTS ==========")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R²:   {r2:.4f}")

evaluate_model("VALIDATION", X_val, y_val)
evaluate_model("TEST", X_test, y_test)

# 6. Display feature importance
print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))

# 7. Save the new model separately
joblib.dump(
    model,
    "results/random_forest_time_features_model.pkl"
)

# 8. Save test predictions
test_predictions = pd.DataFrame({
    "hour": df["hour"].iloc[val_end:].values,
    "actual_cooling_kw": y_test.values,
    "predicted_cooling_kw": model.predict(X_test)
})

test_predictions.to_csv(
    "results/time_features_test_predictions.csv",
    index=False
)

print("\nModel saved successfully.")
print("Predictions saved successfully.")
print("Time-feature experiment completed!")