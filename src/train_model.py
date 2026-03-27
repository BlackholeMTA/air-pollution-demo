import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/merged_features.csv")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "xgb_pm25_model.pkl"
PRED_PATH = OUT_DIR / "predictions.csv"

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "pm25_raw",
    "pm25_lag1",
    "pm25_lag3",
    "pm25_roll6",
    "hour",
    "dayofweek"
]
target_col = "target_pm25"

split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

model = xgb.XGBRegressor(
    booster="gblinear",
    n_estimators=50,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

test_df["pm25_pred_corrected"] = y_pred
test_df.to_csv(PRED_PATH, index=False)
joblib.dump(model, MODEL_PATH)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Saved model to {MODEL_PATH}")
print(f"Saved predictions to {PRED_PATH}")
