
import matplotlib.pyplot as plt
import numpy as np
import os

models = [
    "Random Forest",
    "LightGBM",
    "Polynomial Regression",
    "RF + Time Features",
    "XGBoost"
]

mae = [8.7203, 9.0779, 10.5365, 8.8853, 8.6324]
rmse = [12.3609, 14.0363, 13.9797, 12.4926, 12.1141]

output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

# MAE comparison
plt.figure(figsize=(9, 5))
bars = plt.bar(models, mae)

plt.ylabel("MAE (kW)")
plt.title("Comparison of MAE Across Machine Learning Models")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, mae):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    f"{output_dir}/model_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# RMSE comparison
plt.figure(figsize=(9, 5))
bars = plt.bar(models, rmse)

plt.ylabel("RMSE (kW)")
plt.title("Comparison of RMSE Across Machine Learning Models")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, rmse):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    f"{output_dir}/model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Model comparison charts generated successfully.")
