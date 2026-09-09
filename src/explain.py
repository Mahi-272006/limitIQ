"""
SHAP explainability for the risk model - lets an underwriter/auditor see
WHY a customer got a given risk score, which is required for adverse-action
notices under fair lending regulation (ECOA).
"""
import shap
import pandas as pd
from src.risk_model import load_risk_model
from src.feature_engineering import get_feature_list


def explain_prediction(customer_row: pd.DataFrame):
    model = load_risk_model()
    numeric_feats, cat_feats = get_feature_list()
    all_feats = numeric_feats + cat_feats

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer_row[all_feats])

    contributions = dict(zip(all_feats, shap_values[0] if hasattr(shap_values[0], "__len__") else shap_values))
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    return sorted_contributions

def check_out_of_distribution(customer_df, training_df, numeric_feats, threshold_percentile=1):
    """
    Flags whether a customer's feature values fall outside the range the model
    was actually trained on. If a value is below the 1st percentile or above
    the 99th percentile of training data, predictions for that customer
    should be treated with lower confidence.
    """
    warnings = []
    for feat in numeric_feats:
        low = training_df[feat].quantile(threshold_percentile / 100)
        high = training_df[feat].quantile(1 - threshold_percentile / 100)
        val = customer_df[feat].values[0]
        if val < low or val > high:
            warnings.append(f"{feat} = {val} is outside the typical training range [{low:.2f}, {high:.2f}]")
    return warnings
    
if __name__ == "__main__":
    from src.feature_engineering import load_and_process
    df = load_and_process()
    sample = df.sample(1)
    result = explain_prediction(sample)
    for feat, val in list(result.items())[:10]:
        print(f"{feat}: {val:.4f}")