import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_RAW_PATH = "data/raw/credit_data.csv"
DATA_PROCESSED_PATH = "data/processed/processed_data.csv"
RISK_MODEL_PATH = "models/risk_model.cbm"
UPLIFT_MODEL_PATH = "models/uplift_model.pkl"
CALIBRATION_REPORT_PATH = "models/calibration_report.json"

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./limitiq.db")

# Feature columns used across the project
NUMERIC_FEATURES = [
    "age",
    "income",
    "current_limit",
    "avg_monthly_spend",
    "utilization_rate",
    "months_on_book",
    "num_late_payments_12m",
    "credit_score",
    "debt_to_income",
    "num_credit_inquiries_6m",
]

# Business economics assumptions for expected-value decisioning
PROFIT_MARGIN_ON_SPEND = 0.03      # bank earns ~3% margin (interchange + interest) on extra spend
LOSS_GIVEN_DEFAULT = 0.65          # if a customer defaults, bank typically recovers ~35%, loses 65% of exposure
MAX_ACCEPTABLE_RISK_INCREASE = 0.08  # hard cap: never approve if causal default risk increase exceeds 8%

CATEGORICAL_FEATURES = ["employment_status", "account_type"]

TREATMENT_COL = "limit_increase_treatment"  # 1 = got a limit increase, 0 = didn't
OUTCOME_DEFAULT_COL = "defaulted_next_12m"
OUTCOME_SPEND_COL = "spend_change_pct"