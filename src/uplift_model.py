"""
The core differentiator: estimates the CAUSAL effect of a credit limit increase
on default risk and spend, not just correlation. Uses an X-Learner from EconML,
which handles the fact that treated/untreated customers were NOT randomly assigned
(confounding) via propensity-adjusted meta-learning.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from econml.metalearners import XLearner
from src.feature_engineering import load_and_process, get_feature_list
from src.config import UPLIFT_MODEL_PATH, TREATMENT_COL, OUTCOME_DEFAULT_COL


def prepare_uplift_data(df, numeric_feats, cat_feats):
    df = df.copy()
    # One-hot encode categoricals for econml (it expects numeric arrays)
    df = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    feature_cols = numeric_feats + [c for c in df.columns if any(c.startswith(f) for f in cat_feats)]
    return df, feature_cols


def train_uplift_model():
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    df_enc, feature_cols = prepare_uplift_data(df, numeric_feats, cat_feats)

    X = df_enc[feature_cols].values
    T = df_enc[TREATMENT_COL].values
    Y = df_enc[OUTCOME_DEFAULT_COL].values.astype(float)

    x_learner = XLearner(
        models=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        propensity_model=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    )
    x_learner.fit(Y, T, X=X)

    cate_estimates = x_learner.effect(X)
    print(f"Average causal effect of limit increase on default risk: {cate_estimates.mean():.4f}")
    print(f"Effect range: [{cate_estimates.min():.4f}, {cate_estimates.max():.4f}]")

    with open(UPLIFT_MODEL_PATH, "wb") as f:
        pickle.dump({"model": x_learner, "feature_cols": feature_cols}, f)
    print(f"Uplift model saved -> {UPLIFT_MODEL_PATH}")

    return x_learner, cate_estimates


def load_uplift_model():
    with open(UPLIFT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_causal_effect(customer_df: pd.DataFrame):
    """customer_df must already be feature-engineered + one-hot encoded consistently."""
    bundle = load_uplift_model()
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    for col in feature_cols:
        if col not in customer_df.columns:
            customer_df[col] = 0
    X = customer_df[feature_cols].values
    effect = model.effect(X)
    return effect


if __name__ == "__main__":
    train_uplift_model()