"""
The core differentiator: estimates the CAUSAL effect of a credit limit increase
on default risk and spend, not just correlation. Uses a T-Learner from EconML,
which was benchmarked against X-Learner and DRLearner (see src/compare_uplift_learners.py)
and achieved the best Qini coefficient (0.0095 vs 0.0075 and 0.0009).

The T-Learner trains separate models for treated and control groups,
then estimates the treatment effect as the difference between them.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from econml.metalearners import TLearner
from src.feature_engineering import load_and_process, get_feature_list
from src.config import UPLIFT_MODEL_PATH, TREATMENT_COL, OUTCOME_DEFAULT_COL

# Causal-specific features that help the uplift model detect heterogeneous treatment effects
CAUSAL_FEATURES = [
    "debt_burden_index", "dti_x_utilization", "risk_composite",
    "credit_score_x_dti", "payment_behavior_score",
    "spending_headroom", "utilization_squared", "dti_x_treatment"
]


def prepare_uplift_data(df, numeric_feats, cat_feats):
    df = df.copy()
    # Create treatment interaction before one-hot encoding
    if TREATMENT_COL not in df.columns:
        df[TREATMENT_COL] = 0
    df["dti_x_treatment"] = df["debt_to_income"] * df[TREATMENT_COL]
    # One-hot encode categoricals for econml (it expects numeric arrays)
    df = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    # Exclude identifier, treatment, outcome, and metadata columns from features
    exclude_cols = ["customer_id", TREATMENT_COL, OUTCOME_DEFAULT_COL, "spend_change_pct"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != object]
    return df, feature_cols


def train_uplift_model():
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    df_enc, all_feature_cols = prepare_uplift_data(df, numeric_feats, cat_feats)

    # Ensure all CAUSAL_FEATURES exist
    for feat in CAUSAL_FEATURES:
        if feat not in df_enc.columns:
            df_enc[feat] = 0.0

    X = df_enc[all_feature_cols].values
    T = df_enc[TREATMENT_COL].values
    Y = df_enc[OUTCOME_DEFAULT_COL].values.astype(float)

    # T-Learner: trains separate models for treated and control groups,
    # then estimates CATE = E[Y|T=1, X] - E[Y|T=0, X].
    # Uses GradientBoostingRegressor with moderate capacity to balance
    # learning the heterogeneous treatment effect without overfitting.
    t_learner = TLearner(
        models=GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
    )
    t_learner.fit(Y, T, X=X)

    cate_estimates = t_learner.effect(X)
    print(f"Average causal effect of limit increase on default risk: {cate_estimates.mean():.4f}")
    print(f"Effect range: [{cate_estimates.min():.4f}, {cate_estimates.max():.4f}]")

    with open(UPLIFT_MODEL_PATH, "wb") as f:
        pickle.dump({"model": t_learner, "feature_cols": all_feature_cols}, f)
    print(f"Uplift model saved -> {UPLIFT_MODEL_PATH}")

    return t_learner, cate_estimates


def load_uplift_model():
    with open(UPLIFT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_causal_effect(customer_df: pd.DataFrame):
    """customer_df must already be feature-engineered + one-hot encoded consistently."""
    bundle = load_uplift_model()
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    feature_cols = [c for c in feature_cols if c != "customer_id"]
    for col in feature_cols:
        if col not in customer_df.columns:
            customer_df[col] = 0
    X = customer_df[feature_cols].values
    effect = model.effect(X)
    return effect


if __name__ == "__main__":
    train_uplift_model()
