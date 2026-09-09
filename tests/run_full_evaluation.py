"""
Full model validation suite for LimitIQ. Runs every check we've built:
1. Risk model statistical performance (AUC, precision, recall, calibration)
2. Uplift model causal performance (Qini coefficient)
3. Scenario-based sanity tests across the risk spectrum
4. Out-of-distribution detection on those same scenarios

Produces one consolidated report - this is what you'd hand to a reviewer
(or show in an interview) as evidence the system was properly validated.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.calibration import calibration_curve

from src.risk_model import load_risk_model
from src.calibration import load_platt_calibrator
from src.uplift_model import load_uplift_model, predict_causal_effect, prepare_uplift_data
from src.feature_engineering import load_and_process, get_feature_list, engineer_features, train_test_split_data
from src.explain import check_out_of_distribution
from src.config import (OUTCOME_DEFAULT_COL, TREATMENT_COL,
                         PROFIT_MARGIN_ON_SPEND, LOSS_GIVEN_DEFAULT, MAX_ACCEPTABLE_RISK_INCREASE)
from tests.test_scenarios import SCENARIOS

SECTION = "=" * 60


def section_risk_model(df, numeric_feats, cat_feats):
    print(f"\n{SECTION}\n1. RISK MODEL (Does it rank risky vs safe correctly?)\n{SECTION}")
    model = load_risk_model()
    platt = load_platt_calibrator()
    _, test_df = train_test_split_data(df)

    raw_proba = model.predict_proba(test_df[numeric_feats + cat_feats])[:, 1]
    calibrated_proba = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    threshold = np.percentile(calibrated_proba, 90)
    preds = (calibrated_proba > threshold).astype(int)
    print(f"(Using top-10%-risk threshold: {threshold:.4f}, since base rate is only ~7%)")
    y_true = test_df[OUTCOME_DEFAULT_COL]

    auc = roc_auc_score(y_true, calibrated_proba)
    recall = recall_score(y_true, preds)
    precision = precision_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    prob_true, prob_pred = calibration_curve(y_true, calibrated_proba, n_bins=10)
    calib_gap = np.mean(np.abs(prob_true - prob_pred))

    print(f"ROC-AUC:            {auc:.4f}   {'PASS (>0.6)' if auc > 0.6 else 'FAIL - weak signal'}")
    print(f"Recall:             {recall:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"Calibration gap:    {calib_gap:.4f}   {'PASS (<0.10)' if calib_gap < 0.10 else 'FAIL - needs recalibration'}")
    return {"auc": auc, "recall": recall, "precision": precision, "calibration_gap": calib_gap}


def section_uplift_model(df, numeric_feats, cat_feats):
    print(f"\n{SECTION}\n2. UPLIFT MODEL (Does it find real heterogeneous causal effects?)\n{SECTION}")
    df_enc, feature_cols = prepare_uplift_data(df, numeric_feats, cat_feats)
    bundle = load_uplift_model()
    model = bundle["model"]

    X = df_enc[feature_cols].values
    T = df_enc[TREATMENT_COL].values
    Y = df_enc[OUTCOME_DEFAULT_COL].values
    uplift_scores = model.effect(X)

    order = np.argsort(-uplift_scores)
    y_true, treatment = Y[order], T[order]
    n = len(y_true)
    n_t, n_c = np.cumsum(treatment), np.cumsum(1 - treatment)
    y_t, y_c = np.cumsum(y_true * treatment), np.cumsum(y_true * (1 - treatment))
    qini = (y_t - y_c * (n_t / np.maximum(n_c, 1))) / n
    x = np.arange(1, n + 1) / n
    qini_auc = np.trapz(qini, x) - np.trapz([0, qini[-1]], [0, 1])

    print(f"Qini coefficient:   {qini_auc:.4f}   {'PASS (>0)' if qini_auc > 0 else 'FAIL - no better than random targeting'}")
    print(f"Avg causal effect:  {uplift_scores.mean():.4f}")
    print(f"% customers where limit increase reduces risk: {(uplift_scores < 0).mean()*100:.1f}%")
    print(f"% customers where limit increase raises risk:  {(uplift_scores > 0).mean()*100:.1f}%")
    return {"qini_coefficient": qini_auc}


def section_scenarios(df, numeric_feats, cat_feats):
    print(f"\n{SECTION}\n3. SCENARIO SANITY TESTS (Do individual decisions make sense?)\n{SECTION}")
    model = load_risk_model()
    platt = load_platt_calibrator()
    uplift_bundle = load_uplift_model()

    rows = []
    for profile in SCENARIOS:
        cust_df = pd.DataFrame([profile])
        cust_df = engineer_features(cust_df)

        raw_proba = model.predict_proba(cust_df[numeric_feats + cat_feats])[:, 1][0]
        risk_score = platt.predict_proba([[raw_proba]])[:, 1][0]

        cust_enc, _ = prepare_uplift_data(cust_df, numeric_feats, cat_feats)
        causal_effect = predict_causal_effect(cust_enc)[0]

        ood_warnings = check_out_of_distribution(cust_df, df, numeric_feats)

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

        rows.append({
            "profile": profile["label"],
            "risk_score_%": round(risk_score * 100, 2),
            "causal_effect_%": round(causal_effect * 100, 2),
            "net_value_$": round(net_value, 2),
            "recommendation": rec,
            "out_of_distribution": len(ood_warnings) > 0,
            "num_ood_flags": len(ood_warnings),
        })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    in_dist = results_df[~results_df["out_of_distribution"]]
    scores = in_dist["risk_score_%"].tolist()
    monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    print(f"\nMonotonic risk increase (in-distribution profiles only): {'PASS' if monotonic else 'FAIL'}")
    print(f"Out-of-distribution profiles correctly flagged: {results_df['out_of_distribution'].sum()} of {len(results_df)}")
    return results_df


def main():
    print("LIMITIQ - FULL MODEL VALIDATION REPORT")
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()

    risk_results = section_risk_model(df, numeric_feats, cat_feats)
    uplift_results = section_uplift_model(df, numeric_feats, cat_feats)
    scenario_results = section_scenarios(df, numeric_feats, cat_feats)

    print(f"\n{SECTION}\nSUMMARY\n{SECTION}")
    print(f"Risk model AUC:          {risk_results['auc']:.3f}")
    print(f"Calibration gap:         {risk_results['calibration_gap']:.3f}")
    print(f"Uplift Qini coefficient: {uplift_results['qini_coefficient']:.4f}")
    print(f"Scenario tests run:      {len(scenario_results)}")
    print(f"OOD cases flagged:       {scenario_results['out_of_distribution'].sum()}")


if __name__ == "__main__":
    main()