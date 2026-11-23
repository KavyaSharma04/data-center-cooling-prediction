"""
Data Center Cooling Power Prediction - Deep Learning & Neural Architectures

This module implements state-of-the-art Deep Learning models for data center
cooling power prediction:

1. Tabular ResNet-MLP (Deep Residual Tabular Neural Network)
   - Deep architecture with residual skip connections, Batch Normalization,
     GELU non-linearities, and dropout regularization.
   - Optimized with AdamW and Smooth L1 Loss (Huber loss) for robust gradient dynamics.

2. LSTM Recurrent Sequence Model
   - 2-layer Recurrent Neural Network with 24-hour sliding temporal context window.
   - Captures continuous thermal mass hysteresis and long-range diurnal patterns.

3. Deep Hybrid Stacking Ensemble
   - Optimal blend of Tabular ResNet-MLP and Dynamic GBDT (LightGBM).
   - Combines the smooth inductive bias of deep neural representations with the
     sharp thresholding of gradient boosted trees.

Evaluated on the exact standard chronological benchmark test set:
  - Training:   2016-06-12 to 2023-01-04 (51,587 records)
  - Validation: 2023-01-04 to 2024-05-13 (11,054 records)
  - Testing:    2024-05-14 to 2025-08-29 (11,055 records)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# Ensure directory paths exist
os.makedirs("results/data", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def build_enhanced_features(data_path="results/data/hourly_median_merged_data.csv"):
    """
    Load data and construct psychrometric, thermodynamic, thermal inertia,
    calendar harmonics, and operational feedback features.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df = df.sort_values("hour").reset_index(drop=True)

    # 1. Psychrometric & Thermodynamic Features
    T_f = df["outdoor_air_temp"]
    RH = df["outdoor_air_humidity"].clip(lower=1.0, upper=100.0)
    T_c = (T_f - 32.0) * 5.0 / 9.0

    # Magnus-Tetens Dew Point
    a, b = 17.27, 237.7
    alpha = ((a * T_c) / (b + T_c)) + np.log(RH / 100.0)
    dew_point_c = (b * alpha) / (a - alpha)
    df["dew_point_f"] = (dew_point_c * 9.0 / 5.0) + 32.0

    # Stull Wet-Bulb Temperature
    Tw_c = (
        T_c * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T_c + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    df["wet_bulb_f"] = (Tw_c * 9.0 / 5.0) + 32.0

    df["wet_bulb_depression"] = df["outdoor_air_temp"] - df["wet_bulb_f"]

    # Psychrometric Enthalpy
    es = 0.61078 * np.exp((17.27 * T_c) / (T_c + 237.3))
    e = (RH / 100.0) * es
    p_atm = 101.325
    w_ratio = 0.622 * e / (p_atm - e)
    df["air_enthalpy"] = 1.006 * T_c + w_ratio * (2501.0 + 1.86 * T_c)
    df["vapor_pressure"] = e

    # Economizer & Free-Cooling Proxies
    df["free_cooling_potential"] = np.clip((65.0 - df["wet_bulb_f"]) / 20.0, 0.0, 1.0)
    df["economizer_cutoff"] = (df["wet_bulb_f"] < 55.0).astype(float)
    df["high_chiller_load"] = (df["wet_bulb_f"] > 70.0).astype(float)

    # Cooling Degree Hours
    df["cdh_65"] = np.maximum(0.0, df["outdoor_air_temp"] - 65.0)
    df["cdh_55"] = np.maximum(0.0, df["outdoor_air_temp"] - 55.0)

    # 2. Calendar Harmonics
    hour = df["hour"].dt.hour
    day_of_week = df["hour"].dt.day_of_week
    month = df["hour"].dt.month
    day_of_year = df["hour"].dt.day_of_year

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1) / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1) / 365.25)

    # 3. Thermal Inertia & Autoregressive Dynamics
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"cooling_lag_{lag}"] = df["cooling_kw"].shift(lag)
        df[f"it_lag_{lag}"] = df["it_power_kw"].shift(lag)
        df[f"temp_lag_{lag}"] = df["outdoor_air_temp"].shift(lag)
        df[f"wetbulb_lag_{lag}"] = df["wet_bulb_f"].shift(lag)

    for w in [3, 6, 12, 24]:
        df[f"cooling_roll_mean_{w}"] = df["cooling_kw"].shift(1).rolling(w).mean()
        df[f"cooling_roll_std_{w}"] = df["cooling_kw"].shift(1).rolling(w).std()
        df[f"it_roll_mean_{w}"] = df["it_power_kw"].rolling(w).mean()
        df[f"temp_roll_mean_{w}"] = df["outdoor_air_temp"].rolling(w).mean()

    # Differential rate of change
    df["it_delta_1h"] = df["it_power_kw"] - df["it_power_kw"].shift(1)
    df["temp_delta_1h"] = df["outdoor_air_temp"] - df["outdoor_air_temp"].shift(1)
    df["temp_delta_24h"] = df["outdoor_air_temp"] - df["outdoor_air_temp"].shift(24)

    # Interactions
    df["it_x_temp"] = df["it_power_kw"] * (df["outdoor_air_temp"] / 100.0)
    df["it_x_wetbulb"] = df["it_power_kw"] * (df["wet_bulb_f"] / 100.0)

    df = df.bfill()
    return df


class TabularResNet(nn.Module):
    """
    Deep Tabular ResNet with Batch Normalization, GELU activations,
    Dropout regularization, and Residual Skip Connections.
    """
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.act = nn.GELU()

    def forward(self, x):
        h = self.input_layer(x)
        h = self.act(h + self.block1(h))
        h = self.act(h + self.block2(h))
        return self.head(h).squeeze(-1)


class LSTMRegressor(nn.Module):
    """
    2-Layer Recurrent Neural Network for 24-hour temporal sequential modeling.
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:32s} | MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW | R²: {r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("=" * 65)
    print(" DATA CENTER COOLING POWER PREDICTION: DEEP LEARNING PIPELINE")
    print("=" * 65)

    df = build_enhanced_features()
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    target = "cooling_kw"
    feature_cols = [c for c in df.columns if c not in ["hour", target]]
    print(f"Total features utilized: {len(feature_cols)}")

    X_train = df[feature_cols].iloc[:train_end].values
    y_train = df[target].iloc[:train_end].values

    X_val = df[feature_cols].iloc[train_end:val_end].values
    y_val = df[target].iloc[train_end:val_end].values

    X_test = df[feature_cols].iloc[val_end:].values
    y_test = df[target].iloc[val_end:].values
    # Ensure exact matching timestamp format with benchmark files
    test_timestamps = pd.read_csv("results/median_test_predictions.csv")["hour"].reset_index(drop=True)

    # Scaling
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y, "features": feature_cols}, "results/models/dl_scalers.pkl")

    # =============================================================
    # 1. TRAIN TABULAR RESNET-MLP
    # =============================================================
    print("\nTraining Model 1: Tabular ResNet-MLP...")
    device = torch.device("cpu")
    model_resnet = TabularResNet(X_train_scaled.shape[1], hidden_dim=128).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model_resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_tensor_X = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_scaled, dtype=torch.float32)
    val_tensor_X = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor_X, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    best_val_loss = float('inf')
    best_weights = None
    train_losses, val_losses = [], []

    for epoch in range(25):
        model_resnet.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model_resnet(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model_resnet.eval()
        with torch.no_grad():
            val_pred = model_resnet(val_tensor_X)
            v_loss = criterion(val_pred, val_tensor_y).item()
            val_losses.append(v_loss)

        scheduler.step(v_loss)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = model_resnet.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/25 | Train Loss: {epoch_loss:.4f} | Val Loss: {v_loss:.4f}")

    model_resnet.load_state_dict(best_weights)
    torch.save(model_resnet.state_dict(), "results/models/dl_resnet_model.pt")

    model_resnet.eval()
    with torch.no_grad():
        pred_resnet_scaled = model_resnet(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    pred_resnet = scaler_y.inverse_transform(pred_resnet_scaled.reshape(-1, 1)).flatten()

    # =============================================================
    # 2. TRAIN LSTM SEQUENCE MODEL
    # =============================================================
    print("\nTraining Model 2: LSTM Sequence Regressor (24-Hour Sliding Window)...")
    seq_features = [
        "cooling_kw", "it_power_kw", "outdoor_air_temp", "outdoor_air_humidity",
        "dew_point_f", "wet_bulb_f", "free_cooling_potential", "wet_bulb_depression",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"
    ]
    data_matrix = df[seq_features].values
    target_vector = df["cooling_kw"].values

    scaler_seq = RobustScaler()
    train_seq_scaled = scaler_seq.fit_transform(data_matrix[:train_end])
    val_seq_scaled = scaler_seq.transform(data_matrix[train_end:val_end])
    test_seq_scaled = scaler_seq.transform(data_matrix[val_end:])
    all_seq_scaled = np.vstack([train_seq_scaled, val_seq_scaled, test_seq_scaled])

    SEQ_LEN = 24
    def make_sequences(start_idx, end_idx):
        Xs, ys = [], []
        for i in range(start_idx, end_idx):
            Xs.append(all_seq_scaled[i - SEQ_LEN:i])
            ys.append(y_val_scaled[0] if False else None)
        return np.array(Xs, dtype=np.float32)

    X_train_seq = []
    y_train_seq = []
    for i in range(SEQ_LEN, train_end):
        X_train_seq.append(all_seq_scaled[i - SEQ_LEN:i])
        y_train_seq.append(scaler_y.transform([[target_vector[i]]])[0, 0])
    X_train_seq = np.array(X_train_seq, dtype=np.float32)
    y_train_seq = np.array(y_train_seq, dtype=np.float32)

    X_val_seq = []
    y_val_seq = []
    for i in range(train_end, val_end):
        X_val_seq.append(all_seq_scaled[i - SEQ_LEN:i])
        y_val_seq.append(scaler_y.transform([[target_vector[i]]])[0, 0])
    X_val_seq = np.array(X_val_seq, dtype=np.float32)
    y_val_seq = np.array(y_val_seq, dtype=np.float32)

    X_test_seq = []
    for i in range(val_end, n):
        X_test_seq.append(all_seq_scaled[i - SEQ_LEN:i])
    X_test_seq = np.array(X_test_seq, dtype=np.float32)

    model_lstm = LSTMRegressor(in_features=len(seq_features), hidden_dim=64, num_layers=2)
    optimizer_lstm = torch.optim.AdamW(model_lstm.parameters(), lr=1e-3, weight_decay=1e-4)

    lstm_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
    lstm_loader = DataLoader(lstm_dataset, batch_size=256, shuffle=True)
    val_lstm_X = torch.tensor(X_val_seq)
    val_lstm_y = torch.tensor(y_val_seq)

    best_lstm_loss = float('inf')
    best_lstm_weights = None

    for epoch in range(15):
        model_lstm.train()
        for bx, by in lstm_loader:
            optimizer_lstm.zero_grad()
            l = criterion(model_lstm(bx), by)
            l.backward()
            optimizer_lstm.step()
        
        model_lstm.eval()
        with torch.no_grad():
            v_l = criterion(model_lstm(val_lstm_X), val_lstm_y).item()
        if v_l < best_lstm_loss:
            best_lstm_loss = v_l
            best_lstm_weights = model_lstm.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/15 | Val Loss: {v_l:.4f}")

    model_lstm.load_state_dict(best_lstm_weights)
    torch.save(model_lstm.state_dict(), "results/models/dl_lstm_model.pt")

    model_lstm.eval()
    with torch.no_grad():
        p_lstm_scaled = model_lstm(torch.tensor(X_test_seq)).numpy()
    pred_lstm = scaler_y.inverse_transform(p_lstm_scaled.reshape(-1, 1)).flatten()

    # =============================================================
    # 3. DEEP HYBRID STACKING ENSEMBLE
    # =============================================================
    print("\nConstructing Model 3: Deep Hybrid Stacking Ensemble...")
    # Load dynamic LightGBM predictions
    lgb_pred_df = pd.read_csv("results/improved_dynamic_test_predictions.csv")
    pred_lgb = lgb_pred_df["predicted_cooling_kw"].values

    pred_hybrid = 0.80 * pred_resnet + 0.20 * pred_lgb

    # =============================================================
    # EVALUATIONS & BENCHMARK
    # =============================================================
    print("\n" + "=" * 55)
    print(" DEEP LEARNING TEST EVALUATION RESULTS")
    print("=" * 55)
    res_resnet = evaluate_model(y_test, pred_resnet, "DL Tabular ResNet")
    res_lstm   = evaluate_model(y_test, pred_lstm,   "DL LSTM Sequence Model")
    res_hybrid = evaluate_model(y_test, pred_hybrid, "DL Hybrid Stacking Ensemble")

    # Save Predictions
    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_resnet}).to_csv("results/tables/dl_resnet_test_predictions.csv", index=False)
    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_resnet}).to_csv("results/dl_resnet_test_predictions.csv", index=False)

    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_lstm}).to_csv("results/tables/dl_lstm_test_predictions.csv", index=False)
    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_lstm}).to_csv("results/dl_lstm_test_predictions.csv", index=False)

    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_hybrid}).to_csv("results/tables/dl_hybrid_ensemble_test_predictions.csv", index=False)
    pd.DataFrame({"hour": test_timestamps, "actual_cooling_kw": y_test, "predicted_cooling_kw": pred_hybrid}).to_csv("results/dl_hybrid_ensemble_test_predictions.csv", index=False)

    print("\nSaved predictions and models successfully!")

    # =============================================================
    # GENERATE HIGH-RESOLUTION DIAGNOSTIC FIGURES
    # =============================================================
    sns.set_theme(style="whitegrid")

    # 1. Training Curves
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss (Smooth L1)", color="#3498db", linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", color="#e74c3c", linewidth=2)
    plt.title("Tabular ResNet Training and Validation Loss Curves", fontsize=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dl_training_curves.png", dpi=300)
    plt.close()

    # 2. Actual vs Predicted (Parity Plot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_pred = [
        ("DL Tabular ResNet", pred_resnet, "#2980b9", axes[0]),
        ("DL LSTM (24h Window)", pred_lstm, "#8e44ad", axes[1]),
        ("DL Hybrid Ensemble", pred_hybrid, "#27ae60", axes[2])
    ]
    max_val = max(y_test.max(), pred_resnet.max(), pred_lstm.max())
    for name, p, col, ax in models_pred:
        ax.scatter(y_test, p, alpha=0.3, color=col, s=12)
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal 1:1")
        ax.set_title(f"{name}\nActual vs Predicted", fontsize=11)
        ax.set_xlabel("Actual Cooling Power (kW)")
        ax.set_ylabel("Predicted Cooling Power (kW)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dl_actual_vs_predicted.png", dpi=300)
    plt.close()

    # 3. Residual Error Distribution
    plt.figure(figsize=(10, 5))
    sns.kdeplot(y_test - pred_resnet, label=f"Tabular ResNet (Mean={np.mean(y_test - pred_resnet):.2f})", color="#2980b9", linewidth=2)
    sns.kdeplot(y_test - pred_lstm, label=f"LSTM Sequence (Mean={np.mean(y_test - pred_lstm):.2f})", color="#8e44ad", linewidth=2)
    sns.kdeplot(y_test - pred_hybrid, label=f"Hybrid Ensemble (Mean={np.mean(y_test - pred_hybrid):.2f})", color="#27ae60", linewidth=2)
    plt.axvline(0, color="black", linestyle="--", alpha=0.7)
    plt.title("Deep Learning Error Residual Distributions", fontsize=12)
    plt.xlabel("Prediction Error: Actual - Predicted (kW)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dl_error_residual_distribution.png", dpi=300)
    plt.close()

    # 4. Time Series Zoom (250 hours)
    plt.figure(figsize=(15, 6))
    sl = slice(1200, 1450)
    plt.plot(test_timestamps.iloc[sl], y_test[sl], label="Actual Cooling kW", color="black", linewidth=2)
    plt.plot(test_timestamps.iloc[sl], pred_resnet[sl], label="DL Tabular ResNet", color="#2980b9", linestyle="--", linewidth=1.5)
    plt.plot(test_timestamps.iloc[sl], pred_hybrid[sl], label="DL Hybrid Ensemble", color="#27ae60", linestyle="-.", linewidth=1.8)
    plt.title("Cooling Power Demand Tracking (250-Hour Operational Window)", fontsize=13)
    plt.xlabel("Timestamp")
    plt.ylabel("Cooling Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dl_time_series_zoom.png", dpi=300)
    plt.close()

    # 5. Grand Comparison Summary
    all_models_summary = [
        {"Model": "Baseline Polynomial Reg",       "MAE_kW": 10.5365, "RMSE_kW": 13.9797, "R2": -0.0772, "Family": "Classical"},
        {"Model": "Baseline LightGBM",             "MAE_kW": 9.0779,  "RMSE_kW": 14.0363, "R2": -0.0860, "Family": "Tree"},
        {"Model": "Baseline Time Features (RF)",   "MAE_kW": 8.8853,  "RMSE_kW": 12.4926, "R2": 0.1398,  "Family": "Tree"},
        {"Model": "Baseline Median",               "MAE_kW": 8.7203,  "RMSE_kW": 12.3609, "R2": 0.1578,  "Family": "Baseline"},
        {"Model": "Baseline XGBoost",              "MAE_kW": 8.6324,  "RMSE_kW": 12.1141, "R2": 0.1911,  "Family": "Tree"},
        {"Model": "Physics-Guided Ratio GBDT",     "MAE_kW": 9.2045,  "RMSE_kW": 11.5196, "R2": 0.2686,  "Family": "Physics GBDT"},
        {"Model": "Dynamic Autoregressive GBDT",   "MAE_kW": 3.2829,  "RMSE_kW": 6.2565,  "R2": 0.7842,  "Family": "Dynamic GBDT"},
        {"Model": "DL LSTM Sequence Model",        "MAE_kW": res_lstm["MAE"],   "RMSE_kW": res_lstm["RMSE"],   "R2": res_lstm["R2"],   "Family": "Deep Learning"},
        {"Model": "DL Tabular ResNet",             "MAE_kW": res_resnet["MAE"], "RMSE_kW": res_resnet["RMSE"], "R2": res_resnet["R2"], "Family": "Deep Learning"},
        {"Model": "DL Hybrid Stacking Ensemble",   "MAE_kW": res_hybrid["MAE"], "RMSE_kW": res_hybrid["RMSE"], "R2": res_hybrid["R2"], "Family": "Deep Learning"}
    ]
    df_all = pd.DataFrame(all_models_summary)
    df_all.to_csv("results/tables/comprehensive_model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print(" COMPLETE COMPREHENSIVE PERFORMANCE LEADERBOARD")
    print("=" * 70)
    print(df_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot Full Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pal = ["#bdc3c7", "#bdc3c7", "#bdc3c7", "#bdc3c7", "#bdc3c7", "#3498db", "#2ecc71", "#9b59b6", "#2980b9", "#1abc9c"]
    
    sns.barplot(data=df_all, x="MAE_kW", y="Model", hue="Model", palette=pal, ax=axes[0], legend=False)
    axes[0].set_title("Mean Absolute Error (kW) [Lower is Better]", fontsize=11)
    axes[0].set_xlabel("MAE (kW)")

    sns.barplot(data=df_all, x="RMSE_kW", y="Model", hue="Model", palette=pal, ax=axes[1], legend=False)
    axes[1].set_title("Root Mean Squared Error (kW) [Lower is Better]", fontsize=11)
    axes[1].set_xlabel("RMSE (kW)")

    sns.barplot(data=df_all, x="R2", y="Model", hue="Model", palette=pal, ax=axes[2], legend=False)
    axes[2].set_title("Coefficient of Determination R² [Higher is Better]", fontsize=11)
    axes[2].set_xlabel("R²")

    plt.tight_layout()
    plt.savefig("results/figures/dl_full_benchmark_comparison.png", dpi=300)
    plt.close()

    print("\nSaved all diagnostic figures to results/figures/!")
    print("Deep Learning pipeline completed successfully!")


if __name__ == "__main__":
    main()
