import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PRED_PATH = Path("data/output/predictions_cmaq_hanoi.csv")
OUT_PATH = Path("data/output/comparison_cmaq_hanoi.png")

df = pd.read_csv(PRED_PATH)

plt.figure(figsize=(12, 5))
plt.plot(df["pm25_cmaq_approx"].values, label="CMAQ thô")
plt.plot(df["pm25_pred_corrected"].values, label="Sau AI hiệu chỉnh")
plt.plot(df["pm25_obs"].values, label="Thực tế")
plt.xlabel("Time step")
plt.ylabel("PM2.5")
plt.title("So sánh PM2.5: CMAQ thô vs AI hiệu chỉnh vs Thực tế")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"Saved plot to {OUT_PATH}")
