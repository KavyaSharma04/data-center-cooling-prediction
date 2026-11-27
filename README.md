# 🌡️ Data Center Cooling Power Prediction & Validation

A clean, beginner-friendly machine learning project that predicts how much power data center cooling systems consume, discovers which environmental factors matter most, and validates if models can transfer to unseen supercomputing facilities.

---

## 📌 1. What is this Project About?

Data centers house thousands of servers running the internet, cloud computing, and AI models. All these computers produce immense heat. To prevent hardware from melting, data centers use massive cooling systems (chillers, fans, and pumps).

* **Cooling consumes up to 40% of a data center's total electricity!**
* **The Goal**: Predict cooling power consumption (`cooling_kw`) based on server workload (`it_power_kw`), outdoor temperature (`outdoor_air_temp`), humidity (`outdoor_air_humidity`), and time of day.
* **Why it matters**: Accurate predictions allow operators to dynamically optimize chillers, schedule workloads when outdoor weather is cooler, and save millions of dollars in energy.

---

## 📂 2. Clean Project Structure

We simplified the project to only the essential, high-impact scripts and results:

```text
data-center/
│
├── 📊 Analysis & Experiments:
│   ├── compare_model_errors.py     # Compares all models with MAE, RMSE, R² & statistical Wilcoxon tests
│   ├── run_ablation_study.py       # Tests what happens when you remove weather, time, or IT power
│   └── cross_dataset_validation.py # Tests model transfer on unseen C3 Supercomputer data
│
├── 🤖 Model Training:
│   ├── compare_hourly_aggregation.py # Ingests, cleans, and merges raw server & weather telemetry
│   ├── train_xgboost_model.py        # Baseline tree-based machine learning model
│   ├── train_improved_model.py       # Dynamic Autoregressive GBDT with lag features
│   └── train_deep_learning_model.py  # Deep learning models (LSTM & Tabular ResNet)
│
├── 🖥️ Interactive Dashboard:
│   ├── dashboard.html                # Standalone interactive research dashboard (open in any browser)
│   └── run_dashboard.py              # One-click dashboard launcher (python run_dashboard.py)
│
├── 📁 Data & Results:
│   ├── data/raw/                     # Raw power, weather, and C3 supercomputer data
│   ├── results/data/                 # Clean hourly merged datasets
│   ├── results/figures/              # Visualization graphs and charts
│   ├── results/tables/               # Exported CSV tables and statistical reports
│   └── results/models/               # Trained model weights
│
├── 📄 Documentation:
│   ├── README.md                     # This beginner-friendly guide
│   ├── RESEARCH_PAPER_IEEE_FORMAT.md # Complete formal IEEE research paper
│   └── requirements.txt              # Required Python libraries
```

---

## 💾 3. The Datasets

We work with two real-world datasets:

### Dataset 1: Primary Data Center (Training & In-Domain Testing)
* **Time Span**: June 2016 to August 2025 (over 9 years of hourly data, 73,696 records).
* **Inputs**: Server IT power (kW), outdoor temperature (°C), outdoor humidity (%).
* **Target to Predict**: Cooling power consumption (kW).
* **Train / Test Split**: 70% Training (first 6+ years), 15% Validation, 15% Test (unseen recent 1.3 years).

### Dataset 2: C3 Supercomputing Facility (Cross-Dataset Validation)
* **Source**: Idaho National Laboratory (INL) High Performance Computing Center.
* **Time Span**: July 2020 to January 2026 (48,567 hourly records).
* **Characteristics**: Highly dynamic supercomputing workloads (HPC jobs) with average IT load of 68.3 kW (peaking at 200 kW).

---

## 🏆 4. Pillar 1: Model Comparison

We tested 9 machine learning and deep learning modeling strategies on the held-out test set (11,055 hours).

### Test Set Benchmark Leaderboard:

| Rank | Model Name | Category | MAE (kW) | RMSE (kW) | R² Score | Plain-English Takeaway |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 9 | Polynomial Regression | Classical ML | 10.54 | 13.98 | -0.077 | Polynomial terms explode on unseen workload ranges. |
| 8 | LightGBM Baseline | Standard Tree | 9.08 | 14.04 | -0.086 | Overfits on static inputs; performs worse than median. |
| 7 | Time Features (Random Forest) | Bagging | 8.89 | 12.49 | 0.140 | Time features capture basic diurnal patterns. |
| 6 | Median Baseline | Heuristic | 8.72 | 12.36 | 0.158 | Historical median reference baseline. |
| 5 | XGBoost Baseline | Standard Tree | 8.63 | 12.11 | 0.191 | Standard tree regression; captures non-linear trends. |
| 4 | Dynamic Autoregressive GBDT | Sequential Tree | 3.28 | 6.26 | 0.784 | Uses recent cooling lags; cuts error by over 60%! |
| 3 | DL LSTM Sequence Model | Deep Recurrent | 2.50 | 4.39 | 0.894 | 24-hour sequence memory models thermal delays. |
| 2 | DL Tabular ResNet | Deep Residual | 2.58 | 3.99 | 0.913 | Deep neural network with residual skip connections. |
| **1** | **🏆 DL Hybrid Ensemble** | **Deep Stacking** | **2.48** | **3.82** | **0.920** | **Combines ResNet + Dynamic GBDT for 92% accuracy.** |

> **Key Takeaway**: Standard static machine learning models struggle because data center temperatures depend on thermal delay (heat takes time to build up and dissipate). Incorporating recent sequential history and deep residual learning reduces prediction error by over **71%**!

---

## 🔍 5. Pillar 2: Ablation Study (What Features Matter?)

An **Ablation Study** tests what happens to model accuracy when you deliberately isolate or remove specific feature groups. This quantifies the true marginal predictive power of each telemetry and physics pillar.

While preliminary exploration evaluated a baseline XGBoost model, the comprehensive systematic ablation study is conducted on the project's **state-of-the-art Best Performing Models**: the **Deep Learning Tabular ResNet-MLP**, **Dynamic LightGBM**, and the **DL Hybrid Stacking Ensemble** (80% ResNet + 20% LightGBM) across all 66 engineered features.

### Best Model Systematic Ablation Matrix (11,055 Test Hours, 2024–2025)

| Experiment | Feats | DL ResNet $R^2$ | Dynamic LGBM $R^2$ | DL Hybrid $R^2$ | Hybrid MAE | $\Delta\text{MAE}$ (%) | Physical & Domain Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Best Model** | **66** | **0.9190** | **0.7720** | **0.9270** | **2.26 kW** | **+0.0%** | **SOTA benchmark with all physics, thermal inertia, and closed-loop feedback.** |
| **No Autoregressive Feedback** | 52 | 0.3850 | 0.2325 | 0.3644 | 7.55 kW | +234.0% | **Exogenous Digital Twin Mode**: Pure physical prediction without chiller state feedback. |
| **No Psychrometric Physics** | 49 | 0.9134 | 0.7839 | 0.9145 | 2.21 kW | -2.3% | Ablates wet-bulb, enthalpy & economizers; raw temp/humidity retained. |
| **No Thermal Inertia** | 23 | 0.2142 | 0.2116 | 0.2212 | 8.36 kW | +269.8% | **Thermal memory collapse**: Multi-hour facility heat storage is essential. |
| **No IT Workload Telemetry** | 52 | 0.9364 | 0.7911 | 0.9342 | 1.89 kW | -16.3% | In closed-loop mode, prior cooling state buffers IT load, removing IT shift noise. |
| **No Weather Telemetry** | 34 | 0.9069 | 0.7854 | 0.9156 | 2.54 kW | +12.3% | Outdoor ambient forcing dictates chiller lift and evaporative tower efficiency. |
| **No Calendar Harmonics** | 58 | 0.9135 | 0.7722 | 0.9242 | 2.42 kW | +6.8% | Diurnal sine/cosine harmonics provide fine periodic scheduling corrections. |
| **Raw Baseline Features Only** | 9 | 0.0363 | 0.2250 | 0.0881 | 8.68 kW | +283.7% | **DL on raw inputs fails**: Proves neural networks require dynamic feature engineering. |

### Core Scientific Discoveries:
1. **Dynamic Closed-Loop Dispatch vs Exogenous Digital Twin ($R^2 = 0.9270$ vs $0.3644$)**:
   * Removing past cooling state isolates the gap between capacity planning simulation ($R^2 \approx 0.36$) and real-time 1-hour ahead BMS operational control ($R^2 \approx 0.927$). In real-time dispatch, prior cooling state acts as an unobserved sensor for chiller stage sequencing and pump flow rates.
2. **Facility Thermal Capacitance & Inertial Memory ($+269.8\%$ Error Surge)**:
   * Removing multi-hour lag windows and rolling statistics drops $R^2$ to **0.2212**. Concrete building mass, raised floor plenums, and water loops retain heat across 1–6 hours; static instantaneous models cannot predict thermal transport dynamics.
3. **Deep Residual Learning Requires Feature Engineering**:
   * Tabular ResNet trained on raw unengineered telemetry plummets to $R^2 = 0.0363$. Deep neural architectures alone cannot overcome severe out-of-distribution workload shifts without thermodynamic domain physics and temporal memory.
4. **Graceful Degradation Under Sensor Outages**:
   * If outdoor weather station telemetry completely disconnects, the Hybrid Ensemble still delivers $R^2 = 0.9156$ (only $+12.3\%$ error increase), proving high industrial fault tolerance.

*(Baseline XGBoost ablation results are also preserved in `results/tables/xgboost_ablation_results.csv` for historical reference).*

---

## 🌍 6. Pillar 3: Cross-Dataset Validation (Zero-Shot Transfer)

Can a cooling prediction model trained on one facility transfer to a completely different supercomputing facility without any retraining?

We tested the models on **38,469 hours** of real-world supercomputer workloads from the Idaho National Laboratory (C3) dataset:

1. **Decision Tree Boundary Saturation**:
   * Standard decision trees (LightGBM) partition feature spaces with fixed split thresholds. When queried on unseen supercomputing data where server loads exceed the training maximum, the model routes all inputs into the outermost leaf, flatlining at an invariant **10.40 kW** ($\text{std} = 0.11\text{ kW}$).
2. **Sequential Model Transfer**:
   * Dynamic autoregressive models that leverage sequential lags adapt far better to varying operational states, preserving dynamic responsiveness across operational shifts.

---

## 🚀 7. How to Run the Project (3 Simple Steps)

From `data-center/`:

### Step 1: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 2: Compare All Models & Check Statistics
Runs the statistical benchmark comparing all models with Wilcoxon hypothesis tests:
```powershell
python compare_model_errors.py
```

### Step 3: Run Ablation Study or Cross-Dataset Validation
Run the systematic feature ablation study:
```powershell
# Runs ablation study on the best performing models (Tabular ResNet + LightGBM + Hybrid)
python run_ablation_study.py

# (Optional) Run legacy baseline XGBoost ablation
python run_ablation_study.py --model xgboost
```
Run cross-dataset validation:
```powershell
python cross_dataset_validation.py
```

### Step 4: Launch the Interactive Dashboard
Launch the visual interactive dashboard in your web browser:
```powershell
python run_dashboard.py
```
*(Or simply double-click `dashboard.html` to open it directly in Chrome, Edge, or Firefox!)*

---


