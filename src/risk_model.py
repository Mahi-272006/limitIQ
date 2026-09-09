import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV
from src.feature_engineering import load_and_process, get_feature_list, train_test_split_data
from src.config import RISK_MODEL_PATH, OUTCOME_DEFAULT_COL


def train_risk_model(tune_hyperparameters=True):
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    train_df, test_df = train_test_split_data(df)

    if tune_hyperparameters:
        print("Running hyperparameter search (this takes a few minutes)...")
        param_grid = {
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "iterations": [300, 500, 800],
        }
        base_model = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", verbose=0, random_seed=42
        )
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=15, cv=3,
            scoring="roc_auc", random_state=42, n_jobs=1
        )
        search.fit(
            train_df[numeric_feats + cat_feats], train_df[OUTCOME_DEFAULT_COL],
            cat_features=[train_df[numeric_feats + cat_feats].columns.get_loc(c) for c in cat_feats]
        )
        print(f"Best params: {search.best_params_}")
        print(f"Best CV AUC: {search.best_score_:.4f}")
        best_params = search.best_params_
    else:
        best_params = {"depth": 6, "learning_rate": 0.05, "iterations": 500, "l2_leaf_reg": 3}

    train_pool = Pool(train_df[numeric_feats + cat_feats], train_df[OUTCOME_DEFAULT_COL], cat_features=cat_feats)
    test_pool = Pool(test_df[numeric_feats + cat_feats], test_df[OUTCOME_DEFAULT_COL], cat_features=cat_feats)

    model = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        verbose=100,
        random_seed=42,
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    preds_proba = model.predict_proba(test_pool)[:, 1]
    preds = (preds_proba > 0.5).astype(int)

    auc = roc_auc_score(test_df[OUTCOME_DEFAULT_COL], preds_proba)
    recall = recall_score(test_df[OUTCOME_DEFAULT_COL], preds)
    precision = precision_score(test_df[OUTCOME_DEFAULT_COL], preds)

    print(f"\nFinal Test ROC-AUC: {auc:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

    model.save_model(RISK_MODEL_PATH)
    print(f"Model saved -> {RISK_MODEL_PATH}")
    return model, auc, recall, precision


def load_risk_model():
    model = CatBoostClassifier()
    model.load_model(RISK_MODEL_PATH)
    return model


if __name__ == "__main__":
    train_risk_model(tune_hyperparameters=True)