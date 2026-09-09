"""
Confirms whether the 'Very risky' scenario is out-of-distribution -
i.e., whether the training data actually contains customers like this,
or whether the model is being asked to extrapolate into territory
it never learned from.
"""
import pandas as pd
from src.feature_engineering import load_and_process

df = load_and_process()

print(f"Total training customers: {len(df)}\n")

print("=== Individual factor rarity ===")
print(f"Unemployed customers: {(df['employment_status'] == 'unemployed').sum()} "
      f"({(df['employment_status'] == 'unemployed').mean()*100:.2f}%)")
print(f"Debt-to-income > 0.6: {(df['debt_to_income'] > 0.6).sum()} "
      f"({(df['debt_to_income'] > 0.6).mean()*100:.2f}%)")
print(f"Utilization rate > 1.2: {(df['utilization_rate'] > 1.2).sum()} "
      f"({(df['utilization_rate'] > 1.2).mean()*100:.2f}%)")
print(f"Late payments >= 5: {(df['num_late_payments_12m'] >= 5).sum()} "
      f"({(df['num_late_payments_12m'] >= 5).mean()*100:.2f}%)")
print(f"Months on book <= 4: {(df['months_on_book'] <= 4).sum()} "
      f"({(df['months_on_book'] <= 4).mean()*100:.2f}%)")

print("\n=== Combined profile (all 5 conditions at once) ===")
combined = df[
    (df["employment_status"] == "unemployed")
    & (df["debt_to_income"] > 0.6)
    & (df["utilization_rate"] > 1.2)
    & (df["num_late_payments_12m"] >= 5)
    & (df["months_on_book"] <= 4)
]
print(f"Customers matching ALL 5 extreme conditions together: {len(combined)} "
      f"({len(combined)/len(df)*100:.4f}%)")