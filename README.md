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

## 3. Machine Learning Models

The project includes the following approaches:

- Median Baseline
- Random Forest Regression
- Polynomial Regression
- Time-Feature Model
- LightGBM
- XGBoost

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
│
├── train_median_model.py
├── train_polynomial_regression.py
├── train_time_features_model.py
├── train_lightgbm_model.py
├── train_xgboost_model.py
│
├── results/
│   ├── data/
│   ├── figures/
│   ├── reports/
│   └── tables/
│
├── requirements.txt
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

Run the models from the project root directory:

```powershell
python train_median_model.py
python train_polynomial_regression.py
python train_time_features_model.py
python train_lightgbm_model.py
python train_xgboost_model.py
```

Run analysis and comparison scripts:

```powershell
python compare_model_errors.py
python run_ablation_study.py
python analyze_prediction_errors.py
python analyze_target_distribution.py
python compare_hourly_aggregation.py
```

## 9. Results and Outputs

The project generates:

- Trained machine learning models.
- Test-set predictions.
- Model performance comparison tables.
- Ablation study results.
- Statistical analysis results.
- Data analysis and prediction visualisations.
- Experiment summary reports.

Generated outputs are stored in the `results/` directory.

## 10. Scope and Limitations

The current project focuses on cooling power prediction using the available dataset and implemented machine learning approaches.

Cross-dataset validation is outside the scope of the current project.

## 11. Conclusion

The project investigates machine learning approaches for data center cooling power prediction and compares their performance through regression metrics, ablation experiments, and statistical analysis. The findings provide insights into model performance and the influence of different input features on cooling power prediction.