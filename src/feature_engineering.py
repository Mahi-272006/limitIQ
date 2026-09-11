import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spend_to_income_ratio"] = (df["avg_monthly_spend"] * 12) / df["income"]
    df["limit_to_income_ratio"] = df["current_limit"] / df["income"]
    df["late_payment_rate"] = df["num_late_payments_12m"] / 12
    df["tenure_years"] = df["months_on_book"] / 12
    df["risk_flag_high_utilization"] = (df["utilization_rate"] > 0.8).astype(int)
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"], bins=[0, 580, 670, 740, 850],
        labels=["poor", "fair", "good", "excellent"]
    ).astype(str)
    df["credit_score_x_dti"] = df["credit_score"] * (1 - df["debt_to_income"])
    df["spend_to_monthly_income"] = df["avg_monthly_spend"] / (df["income"] / 12)
    df["inquiries_per_month_on_book"] = df["num_credit_inquiries_6m"] / (df["months_on_book"] + 1)
    df["is_new_account"] = (df["months_on_book"] < 12).astype(int)
    df["high_risk_combo"] = ((df["utilization_rate"] > 0.7) & (df["num_late_payments_12m"] > 0)).astype(int)

    # --- NEW FEATURES for improved risk signal ---
    # Debt burden combines debt-to-income and utilization into a single stress metric
    df["debt_burden_index"] = df["debt_to_income"] * df["utilization_rate"]
    # Interaction of creditworthiness and earning capacity
    df["credit_score_x_income"] = df["credit_score"] * df["income"]
    # Composite payment behavior: late payments + inquiries normalized
    df["payment_behavior_score"] = (df["num_late_payments_12m"] + df["num_credit_inquiries_6m"]) / 12
    # How much spending room is left on the card
    df["spending_headroom"] = 1.0 - df["utilization_rate"]
    # Interaction: debt burden * utilization (double-whammy for high DTI + high util)
    df["dti_x_utilization"] = df["debt_to_income"] * df["utilization_rate"]
    # Composite risk: weighted sum of key risk signals
    df["risk_composite"] = (
        0.3 * (1 - df["credit_score"] / 850)
        + 0.25 * df["debt_to_income"]
        + 0.2 * df["utilization_rate"]
        + 0.15 * df["late_payment_rate"]
        + 0.1 * df["inquiries_per_month_on_book"]
    )
    # Income stability proxy: income relative to current limit (higher = safer)
    df["income_x_tenure"] = df["income"] * df["months_on_book"]
    # Credit utilization squared (captures non-linear risk at high utilization)
    df["utilization_squared"] = df["utilization_rate"] ** 2

    return df


def load_and_process():
    df = pd.read_csv(DATA_RAW_PATH)
    df = engineer_features(df)
    df.to_csv(DATA_PROCESSED_PATH, index=False)
    return df


def get_feature_list():
    return NUMERIC_FEATURES + [
        "spend_to_income_ratio", "limit_to_income_ratio",
        "late_payment_rate", "tenure_years", "risk_flag_high_utilization",
        "credit_score_x_dti", "spend_to_monthly_income",
        "inquiries_per_month_on_book", "is_new_account", "high_risk_combo",
        # NEW features
        "debt_burden_index", "credit_score_x_income", "payment_behavior_score",
        "spending_headroom", "dti_x_utilization", "risk_composite",
        "income_x_tenure", "utilization_squared"
    ], CATEGORICAL_FEATURES + ["credit_score_bucket"]

def train_test_split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["defaulted_next_12m"])


if __name__ == "__main__":
    df = load_and_process()
    print(f"Processed {len(df)} rows -> {DATA_PROCESSED_PATH}")
    print(df.columns.tolist())
