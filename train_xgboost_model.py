import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the median-based dataset
print("Loading median dataset...")

df = pd.read_csv("results/hourly_median_merged_data.csv")

df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

print("Dataset shape:", df.shape)

# 2. Select the same features used by Random Forest
# 2. Create time-based features

df["hour_of_day"] = df["hour"].dt.hour
df["day_of_week"] = df["hour"].dt.dayofweek
df["month"] = df["hour"].dt.month

# Select features, including the new time-based features
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

print("\nFeatures used:", features)

# 3. Chronological split: 70% train, 15% validation, 15% test
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

print("Training period:", df["hour"].iloc[0],
      "to", df["hour"].iloc[train_end - 1])

print("Validation period:", df["hour"].iloc[train_end],
      "to", df["hour"].iloc[val_end - 1])

print("Testing period:", df["hour"].iloc[val_end],
      "to", df["hour"].iloc[-1])

# 4. Configure XGBoost
print("\nTraining XGBoost model...")

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
    early_stopping_rounds=30
)

# 5. Train using validation data for early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Best boosting iteration:", model.best_iteration)

# 6. Evaluate validation and test sets
def evaluate_model(name, X_part, y_part):
    predictions = model.predict(X_part)

    mae = mean_absolute_error(y_part, predictions)
    rmse = np.sqrt(mean_squared_error(y_part, predictions))
    r2 = r2_score(y_part, predictions)

    print(f"\n========== {name} RESULTS ==========")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R²:   {r2:.4f}")

    return predictions

evaluate_model("VALIDATION", X_val, y_val)
test_predictions = evaluate_model("TEST", X_test, y_test)

# 7. Feature importance
print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))

# 8. Save model and predictions separately
joblib.dump(model, "results/xgboost_model.pkl")

prediction_df = pd.DataFrame({
    "hour": df["hour"].iloc[val_end:].values,
    "actual_cooling_kw": y_test.values,
    "predicted_cooling_kw": test_predictions
})

prediction_df.to_csv(
    "results/xgboost_test_predictions.csv",
    index=False
)

print("\nModel saved: results/xgboost_model.pkl")
print("Predictions saved: results/xgboost_test_predictions.csv")
print("\nXGBoost experiment completed!")