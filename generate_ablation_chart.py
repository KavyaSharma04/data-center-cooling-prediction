
import matplotlib.pyplot as plt
import os

configurations = [
    "Full Model",
    "Without Time",
    "Without Weather",
    "Without IT Power"
]

mae = [8.6324, 8.5135, 9.1179, 9.0770]
rmse = [12.1141, 12.1755, 12.9336, 14.0365]

output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

# MAE chart
plt.figure(figsize=(9, 5))
bars = plt.bar(configurations, mae)

plt.ylabel("MAE (kW)")
plt.title("XGBoost Ablation Study - MAE")
plt.xticks(rotation=15, ha="right")
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
    f"{output_dir}/ablation_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# RMSE chart
plt.figure(figsize=(9, 5))
bars = plt.bar(configurations, rmse)

plt.ylabel("RMSE (kW)")
plt.title("XGBoost Ablation Study - RMSE")
plt.xticks(rotation=15, ha="right")
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
    f"{output_dir}/ablation_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Ablation charts generated successfully.")
