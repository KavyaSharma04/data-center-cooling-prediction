import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("results/data/hourly_median_merged_data.csv")
df["hour"] = pd.to_datetime(df["hour"], utc=True)
df = df.sort_values("hour").reset_index(drop=True)

df["hour_of_day"] = df["hour"].dt.hour
df["day_of_week"] = df["hour"].dt.day_of_week
df["month"] = df["hour"].dt.month

features = [
    "it_power_kw",
    "outdoor_air_temp",
    "outdoor_air_humidity",
    "hour_of_day",
    "day_of_week",
    "month",
]
target = "cooling_kw"

# Chronological split: 70% train, 15% validation, 15% test
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = df[features].iloc[:train_end]
y_train = df[target].iloc[:train_end]
X_val = df[features].iloc[train_end:val_end]
y_val = df[target].iloc[train_end:val_end]
X_test = df[features].iloc[val_end:]
y_test = df[target].iloc[val_end:]

# Degree-2 polynomial regression, with feature scaling
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)

model.fit(X_train, y_train)

for name, X_part, y_part in [
    ("VALIDATION", X_val, y_val),
    ("TEST", X_test, y_test),
]:
    predictions = model.predict(X_part)

    print(f"\n========== {name} RESULTS ==========")
    print(f"MAE:  {mean_absolute_error(y_part, predictions):.4f} kW")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_part, predictions)):.4f} kW")
    print(f"R²:   {r2_score(y_part, predictions):.4f}")

test_predictions = model.predict(X_test)

prediction_df = pd.DataFrame({
    "hour": df["hour"].iloc[val_end:].values,
    "actual_cooling_kw": y_test.values,
    "predicted_cooling_kw": test_predictions,
})
prediction_df.to_csv(
    "results/tables/polynomial_regression_test_predictions.csv",
    index=False,
)

print("\nPredictions saved to results/tables/polynomial_regression_test_predictions.csv")