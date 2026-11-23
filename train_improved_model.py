"""
Data Center Cooling Power Prediction - Improved Machine Learning Pipeline

This module implements two state-of-the-art modeling architectures for predicting
data center cooling power consumption:

1. Physics-Guided Cooling Ratio Model (Exogenous / Digital Twin Mode)
   - Predicts cooling power ratio (cooling_kw / it_power_kw) using only exogenous
     operational and psychrometric weather features.
   - Designed for capacity planning, what-if simulations, and out-of-distribution
     extrapolation where cooling telemetry is not known in advance.
   - Solves the out-of-distribution IT power shift (89.3% of test samples exceed
     training IT power max) by predicting scale-invariant cooling intensity.

2. Dynamic Telemetry-Aware Autoregressive Model (Sequential Forecasting Mode)
   - Incorporates facility thermal inertia, psychrometrics, and autoregressive
     chiller operating state feedback (cooling_kw lags and rolling statistics).
   - Designed for real-time operational dispatch, BMS integration, and 1-hour
     ahead cooling demand forecasting.

Both models are evaluated on the exact standard chronological split:
  - Training:   First 70% of chronological records (2016 - Jan 2023)
  - Validation: Next 15% of records (Jan 2023 - May 2024)
  - Testing:    Final 15% of records (May 2024 - Aug 2025)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure output directories exist
os.makedirs("results/data", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def build_psychrometric_and_dynamic_features(data_path="results/data/hourly_median_merged_data.csv"):
    """
    Load data center telemetry and compute advanced psychrometric, thermodynamic,
    thermal inertia, calendar harmonics, and operational features.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df = df.sort_values("hour").reset_index(drop=True)

    print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} raw columns.")

    # -------------------------------------------------------------
    # 1. PSYCHROMETRIC & THERMODYNAMIC FEATURES
    # -------------------------------------------------------------
    # In cooling tower and chiller operations, evaporative capacity is governed
    # by wet-bulb temperature, dew point, and moist air enthalpy.
    T_f = df["outdoor_air_temp"]
    RH = df["outdoor_air_humidity"].clip(lower=1.0, upper=100.0)

    # Convert Fahrenheit to Celsius for thermodynamic equations
    T_c = (T_f - 32.0) * 5.0 / 9.0

    # Magnus-Tetens formula for Dew Point (°C -> °F)
    a = 17.27
    b = 237.7
    alpha = ((a * T_c) / (b + T_c)) + np.log(RH / 100.0)
    dew_point_c = (b * alpha) / (a - alpha)
    df["dew_point_f"] = (dew_point_c * 9.0 / 5.0) + 32.0

    # Stull empirical formula for Wet-Bulb Temperature (°C -> °F)
    Tw_c = (
        T_c * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T_c + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    df["wet_bulb_f"] = (Tw_c * 9.0 / 5.0) + 32.0

    # Wet-bulb depression: difference between dry bulb and wet bulb
    df["wet_bulb_depression"] = df["outdoor_air_temp"] - df["wet_bulb_f"]

    # Psychrometric enthalpy of moist air (kJ/kg)
    es = 0.61078 * np.exp((17.27 * T_c) / (T_c + 237.3))  # Saturation vapor pressure (kPa)
    e = (RH / 100.0) * es                                  # Actual vapor pressure (kPa)
    p_atm = 101.325                                        # Standard sea-level pressure (kPa)
    w_ratio = 0.622 * e / (p_atm - e)                      # Humidity ratio (kg water/kg dry air)
    df["air_enthalpy"] = 1.006 * T_c + w_ratio * (2501.0 + 1.86 * T_c)
    df["vapor_pressure"] = e

    # Economizer & Free-Cooling Operating Proxies
    # Waterside economizers are typically activated when wet-bulb temperature is below ~55°F
    df["free_cooling_potential"] = np.clip((65.0 - df["wet_bulb_f"]) / 20.0, 0.0, 1.0)
    df["economizer_cutoff"] = (df["wet_bulb_f"] < 55.0).astype(float)
    df["high_chiller_load"] = (df["wet_bulb_f"] > 70.0).astype(float)

    # Cooling Degree Hours (CDH)
    df["cdh_65"] = np.maximum(0.0, df["outdoor_air_temp"] - 65.0)
    df["cdh_55"] = np.maximum(0.0, df["outdoor_air_temp"] - 55.0)

    # -------------------------------------------------------------
    # 2. CYCLIC TIME & CALENDAR HARMONICS
    # -------------------------------------------------------------
    hour = df["hour"].dt.hour
    day_of_week = df["hour"].dt.day_of_week
    month = df["hour"].dt.month
    day_of_year = df["hour"].dt.day_of_year

    df["hour_of_day"] = hour
    df["day_of_week"] = day_of_week
    df["month"] = month

    # Sine/Cosine circular encodings
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1) / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1) / 365.25)

    # -------------------------------------------------------------
    # 3. THERMAL INERTIA & OPERATIONAL DYNAMICS
    # -------------------------------------------------------------
    # Exogenous lags: thermal storage and chiller plants exhibit multi-hour delay
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"it_lag_{lag}"] = df["it_power_kw"].shift(lag)
        df[f"temp_lag_{lag}"] = df["outdoor_air_temp"].shift(lag)
        df[f"wetbulb_lag_{lag}"] = df["wet_bulb_f"].shift(lag)

    # Rolling window aggregations
    for w in [3, 6, 12, 24, 48]:
        df[f"it_roll_mean_{w}"] = df["it_power_kw"].rolling(w).mean()
        df[f"temp_roll_mean_{w}"] = df["outdoor_air_temp"].rolling(w).mean()
        df[f"temp_roll_std_{w}"] = df["outdoor_air_temp"].rolling(w).std()

    # Differential rate-of-change (velocities)
    df["it_delta_1h"] = df["it_power_kw"] - df["it_power_kw"].shift(1)
    df["temp_delta_1h"] = df["outdoor_air_temp"] - df["outdoor_air_temp"].shift(1)
    df["temp_delta_24h"] = df["outdoor_air_temp"] - df["outdoor_air_temp"].shift(24)

    # Thermodynamic interaction terms
    df["it_x_temp"] = df["it_power_kw"] * (df["outdoor_air_temp"] / 100.0)
    df["it_x_wetbulb"] = df["it_power_kw"] * (df["wet_bulb_f"] / 100.0)

    # -------------------------------------------------------------
    # 4. AUTOREGRESSIVE TARGET DYNAMICS (FOR SEQUENTIAL FORECASTING)
    # -------------------------------------------------------------
    for lag in [1, 2, 3, 6, 24, 48]:
        df[f"cooling_lag_{lag}"] = df["cooling_kw"].shift(lag)

    for w in [3, 6, 12, 24]:
        df[f"cooling_roll_mean_{w}"] = df["cooling_kw"].shift(1).rolling(w).mean()
        df[f"cooling_roll_std_{w}"] = df["cooling_kw"].shift(1).rolling(w).std()

    # Backfill initial NaN values from rolling/lag shifts
    df = df.bfill()

    # Safe denominator for cooling ratio computation
    safe_it = df["it_power_kw"].clip(lower=50.0)
    df["cooling_ratio"] = df["cooling_kw"] / safe_it

    return df


def evaluate_predictions(y_true, y_pred, model_name, split_name="TEST"):
    """
    Compute MAE, RMSE, and R² for model predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n========== {model_name} [{split_name}] ==========")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R²:   {r2:.4f}")

    return {"Model": model_name, "Split": split_name, "MAE": mae, "RMSE": rmse, "R2": r2}


def plot_diagnostics(y_test, pred_ratio, pred_dynamic, test_timestamps, feature_importance_df):
    """
    Generate visual diagnostic artifacts:
    1. Actual vs Predicted scatter plots
    2. Residual distributions
    3. Time-series zoom comparison
    4. Top 20 feature importances
    """
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted Scatter Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, pred_ratio, alpha=0.3, color="#1f77b4", s=15)
    max_val_0 = max(y_test.max(), pred_ratio.max())
    axes[0].plot([0, max_val_0], [0, max_val_0], "r--", linewidth=2, label="Ideal 1:1")
    axes[0].set_title("Physics-Guided Ratio Model (Exogenous)\nActual vs Predicted", fontsize=12)
    axes[0].set_xlabel("Actual Cooling Power (kW)", fontsize=11)
    axes[0].set_ylabel("Predicted Cooling Power (kW)", fontsize=11)
    axes[0].legend()

    axes[1].scatter(y_test, pred_dynamic, alpha=0.3, color="#2ca02c", s=15)
    max_val_1 = max(y_test.max(), pred_dynamic.max())
    axes[1].plot([0, max_val_1], [0, max_val_1], "r--", linewidth=2, label="Ideal 1:1")
    axes[1].set_title("Dynamic Autoregressive Model (Sequential)\nActual vs Predicted", fontsize=12)
    axes[1].set_xlabel("Actual Cooling Power (kW)", fontsize=11)
    axes[1].set_ylabel("Predicted Cooling Power (kW)", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/figures/improved_actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved: results/figures/improved_actual_vs_predicted.png")

    # 2. Error / Residual Distribution
    plt.figure(figsize=(10, 5))
    err_ratio = y_test - pred_ratio
    err_dynamic = y_test - pred_dynamic

    sns.kdeplot(err_ratio, label=f"Physics-Guided Ratio (Mean={err_ratio.mean():.2f}, Std={err_ratio.std():.2f})", color="#1f77b4", linewidth=2)
    sns.kdeplot(err_dynamic, label=f"Dynamic Autoregressive (Mean={err_dynamic.mean():.2f}, Std={err_dynamic.std():.2f})", color="#2ca02c", linewidth=2)
    plt.axvline(0, color="black", linestyle="--", alpha=0.7)
    plt.title("Error Residual Distributions (Actual - Predicted)", fontsize=13)
    plt.xlabel("Prediction Error (kW)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/improved_error_distribution.png", dpi=300)
    plt.close()
    print("Saved: results/figures/improved_error_distribution.png")

    # 3. Time Series Slice (Zooming into a 300-hour operational test window)
    plt.figure(figsize=(15, 6))
    slice_idx = slice(1000, 1300)
    plt.plot(test_timestamps.iloc[slice_idx], y_test.iloc[slice_idx], label="Actual Cooling kW", color="black", linewidth=1.8)
    plt.plot(test_timestamps.iloc[slice_idx], pred_ratio[slice_idx], label="Physics-Guided Ratio Model", color="#1f77b4", linestyle="--", linewidth=1.5)
    plt.plot(test_timestamps.iloc[slice_idx], pred_dynamic[slice_idx], label="Dynamic Autoregressive Model", color="#2ca02c", linestyle="-.", linewidth=1.5)
    plt.title("Cooling Power Prediction Time Series (300-Hour Test Window)", fontsize=13)
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Cooling Power (kW)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/improved_time_series_overlay.png", dpi=300)
    plt.close()
    print("Saved: results/figures/improved_time_series_overlay.png")

    # 4. Feature Importance Plot (Top 20)
    if feature_importance_df is not None:
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(20)
        sns.barplot(data=top_features, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title("Top 20 Feature Importances (Dynamic Model)", fontsize=13)
        plt.xlabel("Relative Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig("results/figures/improved_feature_importance.png", dpi=300)
        plt.close()
        print("Saved: results/figures/improved_feature_importance.png")


def main():
    print("=" * 60)
    print(" DATA CENTER COOLING POWER PREDICTION: IMPROVED PIPELINE")
    print("=" * 60)

    # 1. Feature Engineering
    df = build_psychrometric_and_dynamic_features()

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    target = "cooling_kw"
    target_ratio = "cooling_ratio"

    # Identify Exogenous Feature Set (Strictly environmental + operational, NO target lags)
    autoregressive_cols = [c for c in df.columns if c.startswith("cooling_lag_") or c.startswith("cooling_roll_")]
    meta_cols = ["hour", target, target_ratio]
    exogenous_features = [c for c in df.columns if c not in meta_cols and c not in autoregressive_cols]

    # Identify Dynamic Feature Set (Exogenous + Autoregressive chiller lags)
    dynamic_features = [c for c in df.columns if c not in meta_cols]

    print(f"\nFeature Summary:")
    print(f"  - Exogenous Features (Pure Physical/Operational): {len(exogenous_features)}")
    print(f"  - Dynamic Features (With Autoregressive State):   {len(dynamic_features)}")

    # Prepare datasets
    X_exo_train = df[exogenous_features].iloc[:train_end]
    X_exo_val   = df[exogenous_features].iloc[train_end:val_end]
    X_exo_test  = df[exogenous_features].iloc[val_end:]

    y_train = df[target].iloc[:train_end]
    y_val   = df[target].iloc[train_end:val_end]
    y_test  = df[target].iloc[val_end:]

    y_r_train = df[target_ratio].iloc[:train_end]
    y_r_val   = df[target_ratio].iloc[train_end:val_end]

    safe_it = df["it_power_kw"].clip(lower=50.0)
    it_test = safe_it.iloc[val_end:].values
    test_timestamps = pd.read_csv("results/median_test_predictions.csv")["hour"].reset_index(drop=True)

    # -------------------------------------------------------------
    # MODEL 1: PHYSICS-GUIDED COOLING RATIO MODEL (EXOGENOUS)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Model 1: Physics-Guided Cooling Ratio Model (Exogenous)...")
    print("=" * 50)

    ratio_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    ratio_model.fit(
        X_exo_train,
        y_r_train,
        eval_set=[(X_exo_val, y_r_val)],
        eval_metric="rmse"
    )

    pred_ratio_val  = ratio_model.predict(X_exo_val) * safe_it.iloc[train_end:val_end].values
    pred_ratio_test = ratio_model.predict(X_exo_test) * it_test

    evaluate_predictions(y_val, pred_ratio_val, "Physics-Guided Ratio Model", "VALIDATION")
    res_ratio = evaluate_predictions(y_test, pred_ratio_test, "Physics-Guided Ratio Model", "TEST")

    # -------------------------------------------------------------
    # MODEL 2: DYNAMIC AUTOREGRESSIVE MODEL (SEQUENTIAL)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Model 2: Dynamic Telemetry-Aware Model (Sequential)...")
    print("=" * 50)

    X_dyn_train = df[dynamic_features].iloc[:train_end]
    X_dyn_val   = df[dynamic_features].iloc[train_end:val_end]
    X_dyn_test  = df[dynamic_features].iloc[val_end:]

    dynamic_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    dynamic_model.fit(
        X_dyn_train,
        y_train,
        eval_set=[(X_dyn_val, y_val)],
        eval_metric="rmse"
    )

    pred_dyn_val  = dynamic_model.predict(X_dyn_val)
    pred_dyn_test = dynamic_model.predict(X_dyn_test)

    evaluate_predictions(y_val, pred_dyn_val, "Dynamic Autoregressive Model", "VALIDATION")
    res_dyn = evaluate_predictions(y_test, pred_dyn_test, "Dynamic Autoregressive Model", "TEST")

    # Feature Importance for Dynamic Model
    feat_imp = pd.DataFrame({
        "Feature": dynamic_features,
        "Importance": dynamic_model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Most Important Features (Dynamic Model):")
    print(feat_imp.head(10).to_string(index=False))

    # -------------------------------------------------------------
    # SAVE PREDICTIONS & MODELS
    # -------------------------------------------------------------
    # 1. Physics-Guided Ratio Model predictions
    df_pred_ratio = pd.DataFrame({
        "hour": test_timestamps.values,
        "actual_cooling_kw": y_test.values,
        "predicted_cooling_kw": pred_ratio_test
    })
    df_pred_ratio.to_csv("results/tables/improved_ratio_test_predictions.csv", index=False)
    df_pred_ratio.to_csv("results/improved_ratio_test_predictions.csv", index=False)

    # 2. Dynamic Model predictions
    df_pred_dyn = pd.DataFrame({
        "hour": test_timestamps.values,
        "actual_cooling_kw": y_test.values,
        "predicted_cooling_kw": pred_dyn_test
    })
    df_pred_dyn.to_csv("results/tables/improved_dynamic_test_predictions.csv", index=False)
    df_pred_dyn.to_csv("results/improved_dynamic_test_predictions.csv", index=False)

    # Save models
    joblib.dump(ratio_model, "results/models/improved_ratio_lgbm.pkl")
    joblib.dump(ratio_model, "results/improved_ratio_lgbm.pkl")
    joblib.dump(dynamic_model, "results/models/improved_dynamic_lgbm.pkl")
    joblib.dump(dynamic_model, "results/improved_dynamic_lgbm.pkl")

    print("\nSaved predictions and models successfully!")

    # -------------------------------------------------------------
    # GENERATE VISUAL DIAGNOSTICS & SUMMARY CHARTS
    # -------------------------------------------------------------
    plot_diagnostics(
        y_test.reset_index(drop=True),
        pred_ratio_test,
        pred_dyn_test,
        test_timestamps,
        feat_imp
    )

    # -------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------
    benchmark_data = [
        {"Model": "Baseline Median",              "MAE_kW": 8.7203, "RMSE_kW": 12.3609, "R2": 0.1578},
        {"Model": "Baseline LightGBM",            "MAE_kW": 9.0779, "RMSE_kW": 14.0363, "R2": -0.0860},
        {"Model": "Baseline Polynomial Reg",      "MAE_kW": 10.5365, "RMSE_kW": 13.9797, "R2": -0.0772},
        {"Model": "Baseline Time Features (RF)",  "MAE_kW": 8.8853, "RMSE_kW": 12.4926, "R2": 0.1398},
        {"Model": "Baseline XGBoost",             "MAE_kW": 8.6324, "RMSE_kW": 12.1141, "R2": 0.1911},
        {"Model": "Physics-Guided Ratio (Ours)",  "MAE_kW": res_ratio["MAE"], "RMSE_kW": res_ratio["RMSE"], "R2": res_ratio["R2"]},
        {"Model": "Dynamic Autoregressive (Ours)","MAE_kW": res_dyn["MAE"],   "RMSE_kW": res_dyn["RMSE"],   "R2": res_dyn["R2"]},
    ]

    summary_df = pd.DataFrame(benchmark_data)
    summary_df.to_csv("results/tables/comprehensive_model_comparison.csv", index=False)

    print("\n" + "=" * 65)
    print(" COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 65)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = ["#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    sns.barplot(data=summary_df, x="MAE_kW", y="Model", hue="Model", palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("Mean Absolute Error (Lower is Better)", fontsize=11)
    axes[0].set_xlabel("MAE (kW)")

    sns.barplot(data=summary_df, x="RMSE_kW", y="Model", hue="Model", palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontsize=11)
    axes[1].set_xlabel("RMSE (kW)")

    sns.barplot(data=summary_df, x="R2", y="Model", hue="Model", palette=palette, ax=axes[2], legend=False)
    axes[2].set_title("Coefficient of Determination R² (Higher is Better)", fontsize=11)
    axes[2].set_xlabel("R²")

    plt.tight_layout()
    plt.savefig("results/figures/improved_models_comparison.png", dpi=300)
    plt.close()
    print("Saved: results/figures/improved_models_comparison.png")

    print("\nImproved model pipeline successfully executed!")


if __name__ == "__main__":
    main()
