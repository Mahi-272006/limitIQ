#!/bin/bash
set -e

# Initialize data and models if they don't exist
if [ ! -f "data/raw/credit_data.csv" ]; then
    echo "Generating training data..."
    python -c "from src.data_simulation import simulate_data; df = simulate_data(50000); df.to_csv('data/raw/credit_data.csv', index=False)"
fi

if [ ! -f "data/processed/processed_data.csv" ]; then
    echo "Running feature engineering..."
    python -c "from src.feature_engineering import load_and_process"
fi

if [ ! -f "models/risk_model.cbm" ]; then
    echo "Training risk model..."
    python -m src.risk_model
fi

if [ ! -f "models/uplift_model.pkl" ]; then
    echo "Training uplift model..."
    python -m src.uplift_model
fi

if [ ! -f "models/platt_calibrator.pkl" ]; then
    echo "Calibrating probabilities..."
    python -m src.calibration
fi

echo "Startup complete. Starting server."
exec "$@"
