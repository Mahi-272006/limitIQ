"""
Simulates a realistic credit-limit-increase dataset since real bank data
isn't publicly available. Creates confounding on purpose (customers who
got limit increases were, historically, lower risk) so the causal model
has something real to correct for.
"""
import numpy as np
import pandas as pd
from src.config import DATA_RAW_PATH

np.random.seed(42)


def simulate_data(n=50000):
    age = np.random.normal(40, 12, n).clip(18, 80)
    income = np.random.lognormal(mean=10.8, sigma=0.4, size=n)
    credit_score = np.random.normal(680, 60, n).clip(300, 850)
    months_on_book = np.random.randint(1, 240, n)
    current_limit = np.random.lognormal(mean=8.5, sigma=0.5, size=n)
    avg_monthly_spend = current_limit * np.random.beta(2, 5, n)
    utilization_rate = (avg_monthly_spend / current_limit).clip(0, 1.5)
    num_late_payments_12m = np.random.poisson(0.6, n)
    debt_to_income = np.random.beta(2, 5, n) * 0.8
    num_credit_inquiries_6m = np.random.poisson(1.2, n)
    employment_status = np.random.choice(
        ["employed", "self_employed", "unemployed", "retired"],
        size=n, p=[0.65, 0.15, 0.08, 0.12]
    )
    account_type = np.random.choice(["standard", "premium", "student"], size=n, p=[0.6, 0.25, 0.15])

    # --- FIX: standardize each component to z-scores BEFORE combining them ---
    # This ensures no single feature (like income, which has a huge raw scale)
    # drowns out the others. Each component now contributes on a comparable scale.
    def z(x):
        return (x - x.mean()) / x.std()

    risk_score = (
        -0.35 * z(credit_score)
        + 0.30 * z(num_late_payments_12m)
        + 0.30 * z(debt_to_income)
        + 0.15 * z(num_credit_inquiries_6m)
        - 0.20 * z(income)
        + np.random.normal(0, 0.5, n)  # irreducible noise
    )

    # Historical treatment assignment: banks tend to give limit increases to LOWER risk customers
    # This creates confounding -> naive comparison of treated vs untreated would be biased
    propensity = 1 / (1 + np.exp(risk_score - np.percentile(risk_score, 60)))
    limit_increase_treatment = np.random.binomial(1, propensity)

    # True causal effect of treatment: increases spend, and has a SMALL true effect on default
    # that varies by customer (heterogeneous treatment effect) - this is what uplift model must find
    true_uplift_effect = -0.10 + 0.35 * debt_to_income # some customers get worse with more credit
    spend_change_pct = (
        5 + limit_increase_treatment * (20 + true_uplift_effect * 50)
        + np.random.normal(0, 8, n)
    )

    # --- FIX: calibrated intercept to target a realistic ~5% base default rate,
    # with risk_score (now properly scaled) driving real separation ---
    default_logit = (
        1.1 * risk_score
        + limit_increase_treatment * true_uplift_effect * 1.5
        - 3.0
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    defaulted_next_12m = np.random.binomial(1, default_prob.clip(0.01, 0.95))

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "age": age.round(0),
        "income": income.round(2),
        "credit_score": credit_score.round(0),
        "months_on_book": months_on_book,
        "current_limit": current_limit.round(2),
        "avg_monthly_spend": avg_monthly_spend.round(2),
        "utilization_rate": utilization_rate.round(3),
        "num_late_payments_12m": num_late_payments_12m,
        "debt_to_income": debt_to_income.round(3),
        "num_credit_inquiries_6m": num_credit_inquiries_6m,
        "employment_status": employment_status,
        "account_type": account_type,
        "limit_increase_treatment": limit_increase_treatment,
        "spend_change_pct": spend_change_pct.round(2),
        "defaulted_next_12m": defaulted_next_12m,
    })
    return df


if __name__ == "__main__":
    df = simulate_data(50000)
    df.to_csv(DATA_RAW_PATH, index=False)
    print(f"Simulated {len(df)} rows -> {DATA_RAW_PATH}")
    print(df.head())
    print("\nTreatment rate:", df["limit_increase_treatment"].mean())
    print("Default rate:", df["defaulted_next_12m"].mean())