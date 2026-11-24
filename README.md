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
│   ├── reproduce_colab_experiment.py # Cleans and merges raw server & weather data
│   ├── train_xgboost_model.py        # Baseline tree-based machine learning model
│   ├── train_improved_model.py       # Dynamic Autoregressive GBDT with lag features
│   └── train_deep_learning_model.py  # Deep learning models (LSTM & Tabular ResNet)
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

An **Ablation Study** tests what happens to model accuracy when you deliberately remove one group of features. This tells us what the model truly cares about.

We tested this systematically on the XGBoost model:

| Experiment | Features Kept | Test MAE | Test R² | What This Proves |
| :--- | :--- | :---: | :---: | :--- |
| **Full Model** | IT Power + Weather + Time | **8.65 kW** | **0.190** | Baseline performance with all information. |
| **No Time Features** | IT Power + Weather | 8.60 kW | 0.164 | Time (hour/month) helps slightly, but is largely captured by weather. |
| **No Weather** | IT Power + Time | 9.15 kW | 0.043 | Removing temperature/humidity degrades model drastically! |
| **No IT Power** | Weather + Time | 9.08 kW | -0.086 | **Without server power, the model completely fails.** |

> **Key Takeaway**: 
> 1. **IT Power** is the primary driver of cooling load.
> 2. **Outdoor Weather** (ambient temperature & humidity) provides the secondary environmental forcing.
> 3. Time features are secondary helpers.

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
Run the feature ablation study:
```powershell
python run_ablation_study.py
```
Run cross-dataset validation:
```powershell
python cross_dataset_validation.py
```

---

## 📜 8. Research Paper

For the full academic paper formatted according to IEEE conference standards (authored by **Kavya Sharma** and **Anushka Singh**), see:
* Markdown source: [RESEARCH_PAPER_IEEE_FORMAT.md](RESEARCH_PAPER_IEEE_FORMAT.md)
