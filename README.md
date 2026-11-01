

# Data Center Cooling Prediction

## Project overview
This project explores machine-learning approaches for predicting data-center cooling power consumption. The current repository includes scripts for training models and analyzing predictions.

## Models
- Median baseline
- Time-feature model
- LightGBM
- XGBoost

## Setup
Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run experiments
Run the desired script from the project root:

```powershell
python train_median_model.py
python train_time_features_model.py
python train_lightgbm_model.py
python train_xgboost_model.py
```

Analysis scripts:

```powershell
python analyze_prediction_errors.py
python analyze_target_distribution.py
python compare_hourly_aggregation.py
python reproduce_colab_experiment.py
```

## Outputs and data
The `data/` and `results/` folders are excluded from Git by `.gitignore`. Add the required dataset locally before running the scripts. Model outputs and predictions are written to `results/`.

## Evaluation requirements
The project evaluation requires:
- Ablation study
- Statistical analysis
- Cross-dataset validation

Document the setup, results, and conclusions for each experiment in the research paper. Ensure the scripts and required data are available so the experiments can be reproduced.

