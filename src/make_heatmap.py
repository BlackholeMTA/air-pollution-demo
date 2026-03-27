import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# giả lập lưới ô nhiễm 20x20
grid = 40 + 20 * np.random.rand(20, 20)

plt.figure(figsize=(8, 6))
plt.imshow(grid, origin="lower", aspect="auto")
plt.colorbar(label="PM2.5")
plt.title("Heatmap nong do PM2.5 (demo)")
plt.xlabel("Longitude index")
plt.ylabel("Latitude index")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap.png", dpi=150)
plt.close()

print("Saved heatmap to data/output/heatmap.png")
