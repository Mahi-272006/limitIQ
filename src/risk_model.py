import pandas as pd
import json
import pickle
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from src.feature_engineering import load_and_process, get_feature_list, train_test_split_data
from src.config import RISK_MODEL_PATH, RISK_MODEL_LIGHTGBM_PATH, MODEL_META_PATH, OUTCOME_DEFAULT_COL


def _train_catboost(train_df, test_df, numeric_feats, cat_feats, params):
    train_pool = Pool(train_df[numeric_feats + cat_feats], train_df[OUTCOME_DEFAULT_COL], cat_features=cat_feats)
    test_pool = Pool(test_df[numeric_feats + cat_feats], test_df[OUTCOME_DEFAULT_COL], cat_features=cat_feats)
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=42,
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
    return model


def _train_lightgbm(train_df, test_df, numeric_feats, cat_feats):
    df_enc = pd.get_dummies(train_df[numeric_feats + cat_feats], columns=cat_feats, drop_first=True)
    df_enc_test = pd.get_dummies(test_df[numeric_feats + cat_feats], columns=cat_feats, drop_first=True)
    for col in df_enc.columns:
        if col not in df_enc_test.columns:
            df_enc_test[col] = 0
    for col in df_enc_test.columns:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[df_enc_test.columns]

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(df_enc, train_df[OUTCOME_DEFAULT_COL])
    return model, df_enc_test


def train_risk_model():
    df = load_and_process()
    numeric_feats, cat_feats = get_feature_list()
    train_df, test_df = train_test_split_data(df)

    # --- CatBoost with good fixed config and new features ---
    print("Training CatBoost with new features...")
    best_params = {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "iterations": 800}
    cat_model = _train_catboost(train_df, test_df, numeric_feats, cat_feats, best_params)
    cat_proba = cat_model.predict_proba(test_df[numeric_feats + cat_feats])[:, 1]
    cat_auc = roc_auc_score(test_df[OUTCOME_DEFAULT_COL], cat_proba)
    print(f"CatBoost Test AUC: {cat_auc:.4f}")
    best_model = cat_model
    best_auc = cat_auc
    best_type = "catboost"

    # --- LightGBM comparison ---
    print("Training LightGBM for comparison...")
    lgb_model, df_enc_test = _train_lightgbm(train_df, test_df, numeric_feats, cat_feats)
    lgb_proba = lgb_model.predict_proba(df_enc_test)[:, 1]
    lgb_auc = roc_auc_score(test_df[OUTCOME_DEFAULT_COL], lgb_proba)
    print(f"LightGBM Test AUC: {lgb_auc:.4f}")

    if lgb_auc > best_auc:
        print(f"LightGBM beats CatBoost! Using LightGBM.")
        best_model = lgb_model
        best_auc = lgb_auc
        best_type = "lightgbm"
    else:
        print(f"CatBoost wins with AUC {best_auc:.4f} vs LightGBM {lgb_auc:.4f}")

    # Save the best model
    if best_type == "catboost":
        best_model.save_model(RISK_MODEL_PATH)
    else:
        with open(RISK_MODEL_LIGHTGBM_PATH, "wb") as f:
            pickle.dump(best_model, f)

    # Save model metadata
    meta = {"model_type": best_type, "auc": round(best_auc, 4)}
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model saved -> {RISK_MODEL_PATH} (type: {best_type})")

    # Final evaluation
    if best_type == "lightgbm":
        preds_proba = best_model.predict_proba(df_enc_test)[:, 1]
    else:
        preds_proba = best_model.predict_proba(test_df[numeric_feats + cat_feats])[:, 1]
    preds = (preds_proba > 0.5).astype(int)
    auc = roc_auc_score(test_df[OUTCOME_DEFAULT_COL], preds_proba)
    recall = recall_score(test_df[OUTCOME_DEFAULT_COL], preds)
    precision = precision_score(test_df[OUTCOME_DEFAULT_COL], preds)
    print(f"\nFinal Test ROC-AUC: {auc:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

    return best_model, auc, recall, precision


def load_risk_model():
    meta = {}
    try:
        with open(MODEL_META_PATH, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        pass

    model_type = meta.get("model_type", "catboost")
    if model_type == "lightgbm":
        with open(RISK_MODEL_LIGHTGBM_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = CatBoostClassifier()
        model.load_model(RISK_MODEL_PATH)
    return model


if __name__ == "__main__":
    train_risk_model()
