from pydantic import BaseModel
from typing import Optional


class CustomerInput(BaseModel):
    customer_id: str
    age: float
    income: float
    credit_score: float
    months_on_book: int
    current_limit: float
    avg_monthly_spend: float
    utilization_rate: float
    num_late_payments_12m: int
    debt_to_income: float
    num_credit_inquiries_6m: int
    employment_status: str
    account_type: str


class DecisionOutput(BaseModel):
    customer_id: str
    baseline_risk_score: float
    causal_effect_on_default: float
    predicted_spend_lift_pct: float
    recommendation: str
    top_reasons: dict


class Token(BaseModel):
    access_token: str
    token_type: str

class DecisionOutput(BaseModel):
    customer_id: str
    baseline_risk_score: float
    causal_effect_on_default: float
    predicted_spend_lift_pct: float
    expected_annual_benefit: float
    expected_annual_cost: float
    net_expected_value: float
    recommendation: str
    top_reasons: dict
    is_out_of_distribution: bool
    ood_warnings: list[str]