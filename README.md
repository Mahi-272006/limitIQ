# LimitIQ - Causal Credit Limit Decisioning System

A ML-powered system that predicts credit risk, estimates the **causal effect** of limit increases on default probability, and provides expected-value-based approve/deny recommendations.

## Architecture

- **Risk Model**: CatBoost classifier with Platt scaling calibration
- **Uplift Model**: EconML T-Learner (benchmarked vs X-Learner and DRLearner)
- **API**: FastAPI with JWT authentication and SQLite audit logging
- **Dashboard**: Streamlit UI
- **Testing**: Full validation suite with scenario sanity checks

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ (for local development)

### Deploy with Docker Compose

```bash
# Copy .env and configure credentials
cp .env.example .env
# Edit .env with your SECRET_KEY, API_USERNAME, API_PASSWORD

# Build and start services
docker-compose up --build
```

Services:
- **API**: `http://localhost:8000` (docs at `http://localhost:8000/docs`)
- **Dashboard**: `http://localhost:8501`

### Build Docker image

```bash
docker build -t limitiq .
```

### Run locally (development)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data and train models
python -m src.data_simulation
python -m src.feature_engineering
python -m src.risk_model
python -m src.uplift_model
python -m src.calibration

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start dashboard (in another terminal)
streamlit run dashboard/app.py --server.port 8501
```

### Run tests

```bash
python -m tests.run_full_evaluation
python -m tests.test_scenarios
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/token` | POST | Get JWT access token |
| `/health` | GET | Health check |
| `/decide-limit-increase` | POST | Evaluate limit increase decision |
| `/audit-log` | GET | View audit log |

## Project Structure

```
src/
  data_simulation.py    # Synthetic data generation (50k rows)
  feature_engineering.py # Feature engineering + preprocessing
  risk_model.py         # CatBoost risk classifier + LightGBM comparison
  uplift_model.py       # EconML T-Learner causal model
  calibration.py        # Platt scaling probability calibration
  explain.py            # SHAP explanations + OOD detection
api/
  main.py               # FastAPI endpoints
  auth.py               # JWT authentication
  database.py           # SQLite audit logging
  schemas.py            # Pydantic models
dashboard/
  app.py                # Streamlit UI
tests/
  run_full_evaluation.py # Full validation suite
  test_scenarios.py      # Scenario sanity tests
```

## Key Metrics

- **Risk Model AUC**: 0.679 (CatBoost with new features)
- **Calibration Gap**: 0.020 (after Platt scaling)
- **Uplift Qini**: 0.0127 (T-Learner, benchmarked vs X-Learner)
- **Recall**: 26.5% (top-10% threshold)
- **OOD Detection**: 4/5 extreme profiles flagged
