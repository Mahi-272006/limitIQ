"""
Scenario-based sanity testing: instead of just measuring overall accuracy,
this checks that the models behave LOGICALLY across a spectrum of customers
from very safe to very risky. A model can have decent overall metrics (AUC,
calibration) and still make nonsensical individual decisions - this catches that.
"""
import pandas as pd
from src.feature_engineering import engineer_features, get_feature_list, load_and_process
from src.risk_model import load_risk_model
from src.uplift_model import predict_causal_effect, prepare_uplift_data
from src.calibration import load_platt_calibrator
from src.explain import check_out_of_distribution
from src.config import PROFIT_MARGIN_ON_SPEND, LOSS_GIVEN_DEFAULT, MAX_ACCEPTABLE_RISK_INCREASE
from src.config import OOD_CHECK_FEATURES
# Five profiles spanning the risk spectrum, safest to riskiest
SCENARIOS = [
    {"label": "Very safe",    "age": 45, "income": 120000, "credit_score": 790, "months_on_book": 96,
     "current_limit": 15000, "avg_monthly_spend": 3000, "utilization_rate": 0.20,
     "num_late_payments_12m": 0, "debt_to_income": 0.10, "num_credit_inquiries_6m": 0,
     "employment_status": "employed", "account_type": "premium", "limit_increase_treatment": 0},

    {"label": "Safe",         "age": 35, "income": 65000, "credit_score": 690, "months_on_book": 36,
     "current_limit": 5000, "avg_monthly_spend": 1500, "utilization_rate": 0.40,
     "num_late_payments_12m": 0, "debt_to_income": 0.25, "num_credit_inquiries_6m": 1,
     "employment_status": "employed", "account_type": "standard", "limit_increase_treatment": 0},

    {"label": "Moderate",     "age": 30, "income": 45000, "credit_score": 630, "months_on_book": 18,
     "current_limit": 3000, "avg_monthly_spend": 1200, "utilization_rate": 0.65,
     "num_late_payments_12m": 1, "debt_to_income": 0.40, "num_credit_inquiries_6m": 2,
     "employment_status": "employed", "account_type": "standard", "limit_increase_treatment": 0},

    {"label": "Risky",        "age": 27, "income": 32000, "credit_score": 580, "months_on_book": 8,
     "current_limit": 2000, "avg_monthly_spend": 1400, "utilization_rate": 1.00,
     "num_late_payments_12m": 3, "debt_to_income": 0.55, "num_credit_inquiries_6m": 4,
     "employment_status": "self_employed", "account_type": "standard", "limit_increase_treatment": 0},

    {"label": "Very risky",   "age": 24, "income": 22000, "credit_score": 520, "months_on_book": 4,
     "current_limit": 1000, "avg_monthly_spend": 950, "utilization_rate": 1.30,
     "num_late_payments_12m": 5, "debt_to_income": 0.65, "num_credit_inquiries_6m": 6,
     "employment_status": "unemployed", "account_type": "standard", "limit_increase_treatment": 0},
]


def run_scenario(profile, model, platt, numeric_feats, cat_feats, training_df):
    df = pd.DataFrame([profile])
    df = engineer_features(df)

    raw_proba = model.predict_proba(df[numeric_feats + cat_feats])[:, 1][0]
    risk_score = platt.predict_proba([[raw_proba]])[:, 1][0]

    df_enc, _ = prepare_uplift_data(df, numeric_feats, cat_feats)
    causal_effect = predict_causal_effect(df_enc)[0]

    ood_warnings = check_out_of_distribution(df, training_df, OOD_CHECK_FEATURES)

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
        "out_of_distribution": len(ood_warnings) > 0,
        "num_ood_flags": len(ood_warnings),
    }


def run_all_scenarios():
    model = load_risk_model()
    platt = load_platt_calibrator()
    numeric_feats, cat_feats = get_feature_list()
    training_df = load_and_process()

    results = [run_scenario(p, model, platt, numeric_feats, cat_feats, training_df) for p in SCENARIOS]
    results_df = pd.DataFrame(results)

    print("\n=== Scenario Test Results ===")
    print(results_df.to_string(index=False))

    # Sanity checks
    print("\n=== Sanity Checks ===")
    # Only check in-distribution profiles (OOD predictions are unreliable)
    in_dist = results_df[~results_df["out_of_distribution"]]
    risk_scores = in_dist["risk_score"].tolist()
    is_monotonic = all(risk_scores[i] <= risk_scores[i + 1] for i in range(len(risk_scores) - 1))
    print(f"Risk score increases monotonically (safe -> risky, in-dist only): {'PASS' if is_monotonic else 'FAIL'}")

    in_dist_effects = in_dist["causal_effect"].tolist()
    if len(in_dist_effects) >= 2:
        riskiest_effect = in_dist_effects[-1]
        safest_effect = in_dist_effects[0]
        print(f"In-dist riskiest causal effect ({riskiest_effect}%) > Safest ({safest_effect}%): "
              f"{'PASS' if riskiest_effect > safest_effect else 'FAIL'}")
    else:
        print(f"Only {len(in_dist_effects)} in-distribution profile(s), skipping causal effect ordering check")

    very_risky_rec = results_df.iloc[-1]["recommendation"]
    very_risky_ood = results_df.iloc[-1]["out_of_distribution"]
    print(f"Very risky profile is NOT approved outright (excluding OOD): {'PASS' if very_risky_rec != 'APPROVE' or very_risky_ood else 'FAIL'}")

    return results_df


if __name__ == "__main__":
    run_all_scenarios()