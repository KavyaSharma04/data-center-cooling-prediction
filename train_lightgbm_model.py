import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the median-based dataset
print("Loading median dataset...")

df = pd.read_csv("results/data/hourly_median_merged_data.csv")
df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

print("Dataset shape:", df.shape)

# 2. Select the same features used by previous models
features = [
    "it_power_kw",
    "outdoor_air_temp",
    "outdoor_air_humidity"
]
target = "cooling_kw"

X = df[features]
y = df[target]

# 3. Chronological split: 70% train, 15% validation, 15% test
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

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

# 4. Configure LightGBM
print("\nTraining LightGBM model...")

model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# 5. Train with validation-based early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        early_stopping(stopping_rounds=50, verbose=False),
        log_evaluation(period=100)
    ]
)

print("Best iteration:", model.best_iteration_)

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

# 8. Save model and test predictions separately
joblib.dump(model, "results/lightgbm_model.pkl")

prediction_df = pd.DataFrame({
    "hour": df["hour"].iloc[val_end:].values,
    "actual_cooling_kw": y_test.values,
    "predicted_cooling_kw": test_predictions
})

prediction_df.to_csv(
    "results/lightgbm_test_predictions.csv",
    index=False
)

print("\nModel saved: results/lightgbm_model.pkl")
print("Predictions saved: results/lightgbm_test_predictions.csv")
print("\nLightGBM experiment completed!")