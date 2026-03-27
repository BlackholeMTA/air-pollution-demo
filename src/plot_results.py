import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PRED_PATH = Path("data/output/predictions.csv")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PRED_PATH)

plt.figure(figsize=(12, 5))
plt.plot(df["pm25_raw"].values[:100], label="Du bao tho")
plt.plot(df["pm25_pred_corrected"].values[:100], label="Sau AI hieu chinh")
plt.plot(df["target_pm25"].values[:100], label="Thuc te")
plt.xlabel("Time step")
plt.ylabel("PM2.5")
plt.title("So sanh PM2.5: Du bao tho vs AI hieu chinh vs Thuc te")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "comparison_plot.png", dpi=150)
plt.close()

print("Saved plot to data/output/comparison_plot.png")
