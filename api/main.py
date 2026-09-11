import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import pandas as pd

from src.calibration import load_platt_calibrator
from src.config import OOD_CHECK_FEATURES

from api.schemas import CustomerInput, DecisionOutput, Token
from api.auth import authenticate_user, create_access_token, get_current_user
from api.database import get_db, DecisionLog
from src.feature_engineering import engineer_features, get_feature_list
from src.risk_model import load_risk_model
from src.uplift_model import predict_causal_effect, prepare_uplift_data
from src.explain import explain_prediction
from src.config import PROFIT_MARGIN_ON_SPEND, LOSS_GIVEN_DEFAULT, MAX_ACCEPTABLE_RISK_INCREASE
from src.explain import check_out_of_distribution
from src.feature_engineering import load_and_process

app = FastAPI(title="LimitIQ - Causal Credit Limit Decisioning API", version="1.0.0")

risk_model = None
_training_df = None
platt_calibrator = None

@app.on_event("startup")
def load_models():
    global risk_model, _training_df, platt_calibrator
    risk_model = load_risk_model()
    _training_df = load_and_process()
    platt_calibrator = load_platt_calibrator()

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decide-limit-increase", response_model=DecisionOutput)

def decide_limit_increase(
    customer: CustomerInput,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    df = pd.DataFrame([customer.dict()])
    df = engineer_features(df)
    numeric_feats, cat_feats = get_feature_list()

    # Determine model type for encoding
    try:
        with open("models/model_meta.json", "r") as f:
            _meta = json.load(f)
        _model_type = _meta.get("model_type", "catboost")
    except FileNotFoundError:
        _model_type = "catboost"

    # One-hot encode for LightGBM
    if _model_type == "lightgbm":
        df_encoded = pd.get_dummies(df[numeric_feats + cat_feats], columns=cat_feats, drop_first=True)
    else:
        df_encoded = df[numeric_feats + cat_feats]

    # Baseline risk score
    raw_risk_score = float(risk_model.predict_proba(df_encoded)[:, 1][0])
    risk_score = float(platt_calibrator.predict_proba([[raw_risk_score]])[:, 1][0])

    # Causal effect estimate
    df_enc, _ = prepare_uplift_data(df, numeric_feats, cat_feats)
    causal_effect = float(predict_causal_effect(df_enc)[0])

    # Simple spend-lift heuristic tied to the causal model's direction (placeholder business rule)
    predicted_spend_lift = 20.0 + (-causal_effect * 30.0)

    # Explanation (uses CatBoost model which handles the raw features)
    shap_result = explain_prediction(df)
    top_reasons = {k: round(float(v), 4) for k, v in list(shap_result.items())[:5]}

    # --- Expected value decisioning ---
    # Benefit: extra yearly spend * bank's profit margin on that spend
    annual_extra_spend = customer.avg_monthly_spend * 12 * (predicted_spend_lift / 100)
    expected_benefit = annual_extra_spend * PROFIT_MARGIN_ON_SPEND

    # Cost: probability of additional default (causal effect) * money at risk if they default
    exposure_at_risk = customer.current_limit
    expected_cost = max(causal_effect, 0) * exposure_at_risk * LOSS_GIVEN_DEFAULT

    net_expected_value = expected_benefit - expected_cost

    # Decision: hard risk cap first, then net value
    if causal_effect > MAX_ACCEPTABLE_RISK_INCREASE:
        recommendation = "deny_limit_increase"
    elif net_expected_value > 0:
        recommendation = "approve_limit_increase"
    elif net_expected_value > -50:  # small negative -> borderline, reduce instead of outright deny
        recommendation = "approve_with_reduced_increase"
    else:
        recommendation = "deny_limit_increase"

    log_entry = DecisionLog(
        customer_id=customer.customer_id,
        requested_by=current_user,
        risk_score=risk_score,
        causal_effect_on_default=causal_effect,
        predicted_spend_lift_pct=predicted_spend_lift,
        decision=recommendation,
        top_reason=list(top_reasons.keys())[0],
    )
    db.add(log_entry)
    db.commit()

    ood_warnings =check_out_of_distribution(df, training_df, OOD_CHECK_FEATURES)
    is_out_of_distribution = len(ood_warnings) > 0
    return DecisionOutput(
        customer_id=customer.customer_id,
        baseline_risk_score=round(risk_score, 4),
        causal_effect_on_default=round(causal_effect, 4),
        predicted_spend_lift_pct=round(predicted_spend_lift, 2),
        expected_annual_benefit=round(expected_benefit, 2),
        expected_annual_cost=round(expected_cost, 2),
        net_expected_value=round(net_expected_value, 2),
        recommendation=recommendation,
        top_reasons=top_reasons,
        is_out_of_distribution=is_out_of_distribution,
        ood_warnings=ood_warnings,
    )


@app.get("/audit-log")
def get_audit_log(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    logs = db.query(DecisionLog).order_by(DecisionLog.timestamp.desc()).limit(100).all()
    return [
        {
            "customer_id": l.customer_id,
            "requested_by": l.requested_by,
            "risk_score": l.risk_score,
            "causal_effect": l.causal_effect_on_default,
            "decision": l.decision,
            "timestamp": l.timestamp,
        }
        for l in logs
    ]