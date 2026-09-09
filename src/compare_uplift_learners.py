"""
Benchmarks multiple causal meta-learners (T-Learner, X-Learner, DR-Learner)
and reports which achieves the best Qini coefficient, so the final choice
is evidence-based rather than arbitrary.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from econml.metalearners import TLearner, XLearner
from econml.dr import DRLearner

from src.feature_engineering import load_and_process, get_feature_list
from src.uplift_model import prepare_uplift_data
from src.config import TREATMENT_COL, OUTCOME_DEFAULT_COL


def compute_qini(y_true, treatment, uplift_scores):
    order = np.argsort(-uplift_scores)
    y_true, treatment = np.array(y_true)[order], np.array(treatment)[order]
    n = len(y_true)
    n_t, n_c = np.cumsum(treatment), np.cumsum(1 - treatment)
    y_t, y_c = np.cumsum(y_true * treatment), np.cumsum(y_true * (1 - treatment))
    qini = (y_t - y_c * (n_t / np.maximum(n_c, 1))) / n
    x = np.arange(1, n + 1) / n
    return np.trapz(qini, x) - np.trapz([0, qini[-1]], [0, 1])


def run_comparison():
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    df_enc, feature_cols = prepare_uplift_data(df, numeric_feats, cat_feats)

    X = df_enc[feature_cols].values
    T = df_enc[TREATMENT_COL].values
    Y = df_enc[OUTCOME_DEFAULT_COL].values.astype(float)

    results = {}

    print("Training T-Learner...")
    t_learner = TLearner(models=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
    t_learner.fit(Y, T, X=X)
    results["T-Learner"] = compute_qini(Y, T, t_learner.effect(X))

    print("Training X-Learner...")
    x_learner = XLearner(
        models=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        propensity_model=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    )
    x_learner.fit(Y, T, X=X)
    results["X-Learner"] = compute_qini(Y, T, x_learner.effect(X))

    print("Training DR-Learner...")
    dr_learner = DRLearner(
        model_propensity=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        random_state=42,
    )
    dr_learner.fit(Y, T, X=X)
    results["DR-Learner"] = compute_qini(Y, T, dr_learner.effect(X))

    print("\n=== Qini Coefficient Comparison ===")
    for name, qini in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {qini:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest performing learner: {best} (Qini = {results[best]:.4f})")
    return results


if __name__ == "__main__":
    run_comparison()