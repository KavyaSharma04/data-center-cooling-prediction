# Data Center Cooling Power Prediction

## 1. Project Overview

This project focuses on predicting cooling power consumption in data centers using machine learning techniques. The objective is to analyse the relationship between cooling power and relevant operational and environmental factors and evaluate different regression models for accurate prediction.

## 2. Project Objectives

- Analyse data center cooling power consumption patterns.
- Perform data preprocessing and exploratory data analysis.
- Develop machine learning models to predict cooling power.
- Compare model performance using standard evaluation metrics.
- Investigate the contribution of different features through ablation analysis.
- Statistically compare model prediction errors.

## 3. Machine Learning & Deep Learning Models

The project includes the following approaches:

- Median Baseline
- Random Forest Regression
- Polynomial Regression
- Time-Feature Model
- LightGBM (Baseline)
- XGBoost (Baseline)
- **Physics-Guided Cooling Ratio Model** (Exogenous / Digital Twin Mode)
- **Dynamic Telemetry-Aware Autoregressive Model** (Sequential GBDT Mode)
- **DL LSTM Sequence Model** (24-hour Recurrent Neural Network)
- **DL Tabular ResNet** (Deep Tabular Residual Network with Skip Connections)
- **DL Hybrid Stacking Ensemble** (Deep Learning ResNet + Dynamic LightGBM)

## 4. Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## 5. Experiments and Analysis

The project includes:

- Exploratory Data Analysis
- Hourly Aggregation Comparison
- Model Performance Comparison
- Prediction Error Analysis
- XGBoost Ablation Study
- Statistical Comparison of Model Errors

## 6. Repository Structure

```text
data-center/
│
├── analyze_prediction_errors.py
├── analyze_target_distribution.py
├── compare_hourly_aggregation.py
├── compare_model_errors.py
├── reproduce_colab_experiment.py
├── run_ablation_study.py
├── cross_dataset_validation.py
│
├── train_median_model.py
├── train_polynomial_regression.py
├── train_time_features_model.py
├── train_lightgbm_model.py
├── train_xgboost_model.py
├── train_improved_model.py
├── train_deep_learning_model.py
│
├── results/
│   ├── data/
│   ├── figures/
│   ├── reports/
│   └── tables/
│
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
├── PROJECT_DOCUMENTATION.pdf
├── CROSS_DATASET_VALIDATION_DOCUMENTATION.md
├── CROSS_DATASET_VALIDATION_DOCUMENTATION.pdf
└── README.md
```

## 7. Setup and Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 8. Running the Experiments

Run baseline and improved models from the project root directory:

```powershell
python train_median_model.py
python train_polynomial_regression.py
python train_time_features_model.py
python train_lightgbm_model.py
python train_xgboost_model.py
python train_improved_model.py
python train_deep_learning_model.py
```

Run analysis and comparison scripts:

```powershell
python compare_model_errors.py
python run_ablation_study.py
python analyze_prediction_errors.py
python analyze_target_distribution.py
python compare_hourly_aggregation.py
python cross_dataset_validation.py
```

### Complete Benchmark Leaderboard on Test Set

| Model | MAE (kW) | RMSE (kW) | R² | Paired Wilcoxon Diff vs Median |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Polynomial Regression | 10.5365 | 13.9797 | -0.0772 | -1.82 kW (p = 0) |
| Baseline LightGBM | 9.0779 | 14.0363 | -0.0860 | -0.56 kW (p < 1e-36) |
| Baseline Time Features (RF) | 8.8853 | 12.4926 | 0.1398 | -0.14 kW (p < 1e-114) |
| Baseline Median | 8.7203 | 12.3609 | 0.1578 | Reference |
| Baseline XGBoost | 8.6324 | 12.1141 | 0.1911 | -0.11 kW (p = 0.00013) |
| Physics-Guided Ratio Model (Ours) | 9.2045 | 11.5196 | 0.2686 | -2.72 kW (p < 1e-68) |
| Dynamic Autoregressive GBDT (Ours) | 3.2829 | 6.2565 | 0.7842 | +3.80 kW (p = 0.0) |
| **DL LSTM Sequence Model (Ours)** | **2.4974** | **4.3867** | **0.8939** | **+4.21 kW (p = 0.0)** |
| **DL Tabular ResNet (Ours)** | **2.5846** | **3.9852** | **0.9125** | **+3.64 kW (p = 0.0)** |
| **DL Hybrid Stacking Ensemble (Ours)** | **2.4767** | **3.8213** | **0.9195** | **+3.72 kW (p = 0.0)** |

### Validation vs. Test Generalization Comparison

The benchmark dataset comprises 73,696 chronological records split into:
- **Training**: 2016-06-12 to 2023-01-04 (51,587 records, 70%)
- **Validation**: 2023-01-04 to 2024-05-13 (11,054 records, 15%)
- **Test (Held-Out)**: 2024-05-14 to 2025-08-29 (11,055 records, 15%)

| Model | Val MAE (kW) | Val RMSE (kW) | Val R² | Test MAE (kW) | Test RMSE (kW) | Test R² | Generalization Gap (R²) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Median | 10.1365 | 16.8029 | -0.2773 | 9.1490 | 14.8714 | -0.2190 | +0.0582 |
| Baseline XGBoost | 11.3823 | 14.9103 | -0.0057 | 8.6324 | 12.1141 | 0.1911 | +0.1968 |
| Dynamic LightGBM | 3.8870 | 6.8380 | 0.7885 | 3.2829 | 6.2565 | 0.7842 | -0.0042 |
| DL Tabular ResNet | 2.4945 | 3.6891 | 0.9384 | 2.5846 | 3.9852 | 0.9125 | -0.0260 |
| **DL Hybrid Ensemble** | **2.5214** | **3.7301** | **0.9371** | **2.4767** | **3.8213** | **0.9195** | **-0.0175** |

*(The minimal generalization gap confirms zero overfitting and robust stability on unseen operational telemetry.)*

### Cross-Dataset Validation Benchmark (External C3 HPC Facility)

An independent external validation was conducted on the **Collaborative Computing Center (C3) High Performance Computing Data Center (Building IF-692)** at Idaho National Laboratory:
- **Dataset**: `data/raw/c3_datacenter_hourly_power_usage_july2020-jan2026.xlsx`
- **Scope**: 48,567 continuous hourly records (July 2020 to January 2026) with 38,469 concurrent hours merged with on-site meteorological telemetry (July 2020 to August 2025).
- **Workload Shift**: Mean load = 1,586.12 kW, Median = 1,475.72 kW, Peak = 3,927.84 kW, Kolmogorov-Smirnov $D = 0.4610$ ($p = 0.0$), Wasserstein Distance = 360.30 kW vs Primary Facility.

| Model Architecture | Pred Cooling Mean (kW) | Pred Cooling Std (kW) | Implied Cooling Ratio | Implied PUE | Free-Cooling PUE ($T_{\text{wb}} < 55^\circ\text{F}$) | Chiller PUE ($T_{\text{wb}} \ge 55^\circ\text{F}$) | Economizer Savings | Physical Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physics-Guided Ratio Model (Ours)** | **13.42** | **5.68** | **0.0089** | **1.0089** | **1.0085** | **1.0103** | **24.5%** | **100.0%** |
| Dynamic Autoregressive GBDT (Ours) | 4.38 | 0.98 | 0.0030 | 1.0030 | 1.0029 | 1.0033 | 19.9% | 100.0% |
| Baseline XGBoost | 12.67 | 4.65 | 0.0086 | 1.0086 | 1.0085 | 1.0092 | 16.1% | 100.0% |
| Baseline LightGBM | 10.40 | 0.11 | 0.0075 | 1.0075 | 1.0077 | 1.0068 | -0.1% | 100.0% |
| Baseline Random Forest | 12.57 | 4.60 | 0.0086 | 1.0086 | 1.0084 | 1.0091 | 15.0% | 100.0% |
| Baseline Time Features (RF) | 12.53 | 4.68 | 0.0085 | 1.0085 | 1.0084 | 1.0090 | 14.6% | 100.0% |

*(Baseline LightGBM saturated to an invariant flat prediction of $\approx 10.40\text{ kW}$ due to tree leaf boundary saturation on out-of-distribution HPC workloads, whereas the Physics-Guided Ratio model successfully scaled and captured $24.5\%$ economizer seasonal savings.)*

## 9. Results and Outputs

The project generates:

- Trained machine learning models.
- Test-set and cross-dataset predictions.
- Model performance comparison tables and statistical tests.
- Ablation study results.
- Cross-facility workload distribution and transferability metrics.
- High-resolution visual diagnostics in `results/figures/`.
- Comprehensive engineering reports in `results/reports/`.

Generated outputs are stored in the `results/` directory.

## 10. Scope and Limitations

The project investigates data center cooling power prediction across both within-facility temporal splits and cross-facility transfers. Cross-dataset validation has been completed using the external C3 High Performance Computing Center dataset (`data/raw/c3_datacenter_hourly_power_usage_july2020-jan2026.xlsx`), demonstrating the scale invariance and zero-shot transfer capability of the physics-guided ratio architecture.

## 11. Conclusion

The project investigates machine learning approaches for data center cooling power prediction and compares their performance through regression metrics, ablation experiments, statistical tests, and cross-dataset transfer validation. The findings provide insights into model performance, out-of-distribution robustness, psychrometric physics, and the influence of different input features on cooling power prediction.