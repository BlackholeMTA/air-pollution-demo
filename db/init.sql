CREATE TABLE IF NOT EXISTS pm25_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    station_id VARCHAR(50),
    pm25_raw FLOAT,
    pm25_obs FLOAT,
    pm25_lag1 FLOAT,
    pm25_lag3 FLOAT,
    pm25_roll6 FLOAT,
    hour INT,
    dayofweek INT,
    target_pm25 FLOAT,
    pm25_pred_corrected FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
