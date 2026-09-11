import streamlit as st
import requests
import pandas as pd
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LimitIQ", layout="wide")
st.title("LimitIQ — Causal Credit Limit Decisioning")
st.caption("Not just 'is this customer risky' — 'what happens if we raise their limit?'")

if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    st.subheader("Login")
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", type="password", value="admin123")
    if st.button("Login"):
        resp = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if resp.status_code == 200:
            st.session_state.token = resp.json()["access_token"]
            st.success("Logged in")
        else:
            st.error("Login failed")

if st.session_state.token:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

    st.header("Customer Evaluation")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.text_input("Customer ID", "CUST_000123")
        age = st.number_input("Age", 18, 90, 35)
        income = st.number_input("Annual Income (₹)", 10000, 500000, 65000)
        credit_score = st.number_input("Credit Score", 300, 850, 690)
    with col2:
        months_on_book = st.number_input("Months on Book", 1, 300, 36)
        current_limit = st.number_input("Current Limit (₹)", 500, 100000, 5000)
        avg_monthly_spend = st.number_input("Avg Monthly Spend (₹)", 0, 50000, 1500)
        utilization_rate = st.slider("Utilization Rate", 0.0, 1.5, 0.4)
    with col3:
        num_late_payments_12m = st.number_input("Late Payments (12m)", 0, 12, 0)
        debt_to_income = st.slider("Debt-to-Income", 0.0, 0.8, 0.25)
        num_credit_inquiries_6m = st.number_input("Credit Inquiries (6m)", 0, 10, 1)
        employment_status = st.selectbox("Employment Status", ["employed", "self_employed", "unemployed", "retired"])
        account_type = st.selectbox("Account Type", ["standard", "premium", "student"])

    if st.button("Evaluate Limit Increase", type="primary"):
        payload = {
            "customer_id": customer_id, "age": age, "income": income,
            "credit_score": credit_score, "months_on_book": months_on_book,
            "current_limit": current_limit, "avg_monthly_spend": avg_monthly_spend,
            "utilization_rate": utilization_rate, "num_late_payments_12m": num_late_payments_12m,
            "debt_to_income": debt_to_income, "num_credit_inquiries_6m": num_credit_inquiries_6m,
            "employment_status": employment_status, "account_type": account_type,
        }
        resp = requests.post(f"{API_URL}/decide-limit-increase", json=payload, headers=headers)
        if resp.status_code == 200:
            result = resp.json()
            st.subheader("Decision")
            if result.get("is_out_of_distribution"):
                st.warning("⚠️ This customer's profile is unusual compared to the training data. "
                           "Model confidence may be lower than normal — recommend manual review.")
                with st.expander("See which factors are out of range"):
                    for w in result["ood_warnings"]:
                        st.write(f"- {w}")
                        
            rec_color = {"approve_limit_increase": "green", "approve_with_reduced_increase": "orange", "deny_limit_increase": "red"}
            st.markdown(f"**Recommendation:** :{rec_color.get(result['recommendation'],'blue')}[{result['recommendation'].replace('_',' ').title()}]")

            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline Risk Score", f"{result['baseline_risk_score']:.2%}")
            m2.metric("Causal Effect on Default", f"{result['causal_effect_on_default']:+.2%}")
            m3.metric("Predicted Spend Lift", f"{result['predicted_spend_lift_pct']:.1f}%")

            st.subheader("Expected Value Analysis")
            e1, e2, e3 = st.columns(3)
            e1.metric("Expected Annual Benefit", f"₹{result['expected_annual_benefit']:,.2f}")
            e2.metric("Expected Annual Cost", f"₹{result['expected_annual_cost']:,.2f}")
            e3.metric("Net Expected Value", f"₹{result['net_expected_value']:,.2f}",
                    delta="Positive" if result['net_expected_value'] > 0 else "Negative")

            st.subheader("Top Factors (SHAP)")
            reasons_df = pd.DataFrame(list(result["top_reasons"].items()), columns=["Feature", "Impact"])
            st.bar_chart(reasons_df.set_index("Feature"))
        else:
            st.error(f"Error: {resp.text}")

    st.header("Audit Log")
    if st.button("Refresh Audit Log"):
        resp = requests.get(f"{API_URL}/audit-log", headers=headers)
        if resp.status_code == 200:
            st.dataframe(pd.DataFrame(resp.json()))
else:
    st.info("Please log in from the sidebar to continue.")