"""
Scenario-based sanity testing: instead of just measuring overall accuracy,
this checks that the models behave LOGICALLY across a spectrum of customers
from very safe to very risky. A model can have decent overall metrics (AUC,
calibration) and still make nonsensical individual decisions - this catches that.
"""
import pandas as pd
from src.feature_engineering import engineer_features, get_feature_list
from src.risk_model import load_risk_model
from src.uplift_model import predict_causal_effect, prepare_uplift_data
from src.calibration import load_platt_calibrator
from src.config import PROFIT_MARGIN_ON_SPEND, LOSS_GIVEN_DEFAULT, MAX_ACCEPTABLE_RISK_INCREASE

# Five profiles spanning the risk spectrum, safest to riskiest
SCENARIOS = [
    {"label": "Very safe",    "age": 45, "income": 120000, "credit_score": 790, "months_on_book": 96,
     "current_limit": 15000, "avg_monthly_spend": 3000, "utilization_rate": 0.20,
     "num_late_payments_12m": 0, "debt_to_income": 0.10, "num_credit_inquiries_6m": 0,
     "employment_status": "employed", "account_type": "premium"},

    {"label": "Safe",         "age": 35, "income": 65000, "credit_score": 690, "months_on_book": 36,
     "current_limit": 5000, "avg_monthly_spend": 1500, "utilization_rate": 0.40,
     "num_late_payments_12m": 0, "debt_to_income": 0.25, "num_credit_inquiries_6m": 1,
     "employment_status": "employed", "account_type": "standard"},

    {"label": "Moderate",     "age": 30, "income": 45000, "credit_score": 630, "months_on_book": 18,
     "current_limit": 3000, "avg_monthly_spend": 1200, "utilization_rate": 0.65,
     "num_late_payments_12m": 1, "debt_to_income": 0.40, "num_credit_inquiries_6m": 2,
     "employment_status": "employed", "account_type": "standard"},

    {"label": "Risky",        "age": 27, "income": 32000, "credit_score": 580, "months_on_book": 8,
     "current_limit": 2000, "avg_monthly_spend": 1400, "utilization_rate": 1.00,
     "num_late_payments_12m": 3, "debt_to_income": 0.55, "num_credit_inquiries_6m": 4,
     "employment_status": "self_employed", "account_type": "standard"},

    {"label": "Very risky",   "age": 24, "income": 22000, "credit_score": 520, "months_on_book": 4,
     "current_limit": 1000, "avg_monthly_spend": 950, "utilization_rate": 1.30,
     "num_late_payments_12m": 5, "debt_to_income": 0.65, "num_credit_inquiries_6m": 6,
     "employment_status": "unemployed", "account_type": "standard"},
]


def run_scenario(profile, model, platt, numeric_feats, cat_feats):
    df = pd.DataFrame([profile])
    df = engineer_features(df)

    raw_proba = model.predict_proba(df[numeric_feats + cat_feats])[:, 1][0]
    risk_score = platt.predict_proba([[raw_proba]])[:, 1][0]

    df_enc, _ = prepare_uplift_data(df, numeric_feats, cat_feats)
    causal_effect = predict_causal_effect(df_enc)[0]

    spend_lift = 20.0 + (-causal_effect * 30.0)
    annual_extra_spend = profile["avg_monthly_spend"] * 12 * (spend_lift / 100)
    expected_benefit = annual_extra_spend * PROFIT_MARGIN_ON_SPEND
    expected_cost = max(causal_effect, 0) * profile["current_limit"] * LOSS_GIVEN_DEFAULT
    net_value = expected_benefit - expected_cost

    if causal_effect > MAX_ACCEPTABLE_RISK_INCREASE:
        rec = "DENY"
    elif net_value > 0:
        rec = "APPROVE"
    elif net_value > -50:
        rec = "REDUCE"
    else:
        rec = "DENY"

    return {
        "profile": profile["label"],
        "risk_score": round(risk_score * 100, 2),
        "causal_effect": round(causal_effect * 100, 2),
        "net_value": round(net_value, 2),
        "recommendation": rec,
    }


def run_all_scenarios():
    model = load_risk_model()
    platt = load_platt_calibrator()
    numeric_feats, cat_feats = get_feature_list()

    results = [run_scenario(p, model, platt, numeric_feats, cat_feats) for p in SCENARIOS]
    results_df = pd.DataFrame(results)

    print("\n=== Scenario Test Results ===")
    print(results_df.to_string(index=False))

    # Sanity checks
    print("\n=== Sanity Checks ===")
    risk_scores = results_df["risk_score"].tolist()
    is_monotonic = all(risk_scores[i] <= risk_scores[i + 1] for i in range(len(risk_scores) - 1))
    print(f"Risk score increases monotonically (safe -> risky): {'PASS' if is_monotonic else 'FAIL'}")

    causal_effects = results_df["causal_effect"].tolist()
    riskiest_effect = causal_effects[-1]
    safest_effect = causal_effects[0]
    print(f"Riskiest profile causal effect ({riskiest_effect}%) > Safest profile causal effect ({safest_effect}%): "
          f"{'PASS' if riskiest_effect > safest_effect else 'FAIL'}")

    very_risky_rec = results_df.iloc[-1]["recommendation"]
    print(f"Very risky profile is NOT approved outright: {'PASS' if very_risky_rec != 'APPROVE' else 'FAIL'}")

    return results_df


if __name__ == "__main__":
    run_all_scenarios()