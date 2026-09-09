"""
Checks whether predicted probabilities match observed frequencies.
Also fits and SAVES a Platt scaling model to correct for the distortion
introduced by class-balanced training (which improves recall but skews
raw probabilities) - so this correction can be applied in the live API too.
"""
import json
import pickle
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from src.risk_model import load_risk_model
from src.feature_engineering import load_and_process, get_feature_list, train_test_split_data
from src.config import CALIBRATION_REPORT_PATH, OUTCOME_DEFAULT_COL

PLATT_MODEL_PATH = "models/platt_calibrator.pkl"


def compute_gap(y_true, proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins)
    gap = float(np.mean(np.abs(prob_true - prob_pred)))
    return gap, prob_true, prob_pred


def run_calibration_check():
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    train_df, test_df = train_test_split_data(df)

    model = load_risk_model()
    raw_train_proba = model.predict_proba(train_df[numeric_feats + cat_feats])[:, 1]
    raw_test_proba = model.predict_proba(test_df[numeric_feats + cat_feats])[:, 1]

    gap_before, _, _ = compute_gap(test_df[OUTCOME_DEFAULT_COL], raw_test_proba)

    # Fit Platt scaling on TRAINING data
    platt = LogisticRegression()
    platt.fit(raw_train_proba.reshape(-1, 1), train_df[OUTCOME_DEFAULT_COL])

    # SAVE the fitted calibrator so the API can reuse it
    with open(PLATT_MODEL_PATH, "wb") as f:
        pickle.dump(platt, f)
    print(f"Platt calibrator saved -> {PLATT_MODEL_PATH}")

    # Evaluate on TEST data
    calibrated_test_proba = platt.predict_proba(raw_test_proba.reshape(-1, 1))[:, 1]
    gap_after, prob_true, prob_pred = compute_gap(test_df[OUTCOME_DEFAULT_COL], calibrated_test_proba)

    report = {
        "calibration_gap_before_platt_scaling": round(gap_before, 4),
        "calibration_gap_after_platt_scaling": round(gap_after, 4),
        "bins_after_calibration": [
            {"predicted": round(float(p), 4), "actual": round(float(a), 4)}
            for p, a in zip(prob_pred, prob_true)
        ],
    }
    with open(CALIBRATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Calibration gap BEFORE Platt scaling: {gap_before:.4f}")
    print(f"Calibration gap AFTER Platt scaling:  {gap_after:.4f}")
    print(f"Report saved -> {CALIBRATION_REPORT_PATH}")
    return report


def load_platt_calibrator():
    with open(PLATT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    run_calibration_check()