"""
Generates a synthetic but realistic telecom customer churn dataset.

The churn label is created from a logistic function of several underlying
features plus noise, so the resulting dataset has genuine, learnable
signal (similar in spirit to the classic public "Telco Customer Churn"
dataset), while being fully reproducible offline.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 4000

# --- Core customer attributes -------------------------------------------------
tenure_months = np.random.gamma(shape=2.0, scale=15, size=N).clip(0, 72).round().astype(int)

contract_type = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    size=N,
    p=[0.55, 0.25, 0.20],
)

monthly_charges = np.round(
    np.random.normal(loc=65, scale=25, size=N).clip(15, 150), 2
)

# Total charges roughly tracks tenure * monthly charges with some noise
total_charges = np.round(
    monthly_charges * tenure_months * np.random.normal(1.0, 0.05, size=N) + np.random.normal(0, 20, size=N),
    2,
).clip(0, None)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.45, 0.20]
)

tech_support = np.random.choice(["Yes", "No"], size=N, p=[0.35, 0.65])
online_security = np.random.choice(["Yes", "No"], size=N, p=[0.30, 0.70])
paperless_billing = np.random.choice(["Yes", "No"], size=N, p=[0.6, 0.4])

payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=N,
    p=[0.35, 0.20, 0.225, 0.225],
)

num_support_calls = np.random.poisson(lam=1.5, size=N).clip(0, 12)

senior_citizen = np.random.choice([0, 1], size=N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=N, p=[0.30, 0.70])

age = np.round(np.random.normal(42, 15, size=N).clip(18, 85)).astype(int)

# --- Build churn probability from a linear combination of signal features ----
contract_risk = np.select(
    [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
    [1.1, -0.4, -1.2],
)
internet_risk = np.select(
    [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
    [0.55, -0.1, -0.5],
)
support_risk = np.where(tech_support == "No", 0.35, -0.35)
security_risk = np.where(online_security == "No", 0.30, -0.30)
payment_risk = np.where(payment_method == "Electronic check", 0.30, -0.10)

logit = (
    -1.1
    + contract_risk
    + internet_risk
    + support_risk
    + security_risk
    + payment_risk
    + 0.18 * num_support_calls
    - 0.02 * tenure_months
    + 0.01 * (monthly_charges - 65) / 10
    + 0.15 * senior_citizen
    + np.random.normal(0, 0.6, size=N)  # noise
)

churn_prob = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(N) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame(
    {
        "customer_id": [f"CUST-{i:05d}" for i in range(N)],
        "age": age,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": num_support_calls,
        "churn": churn_label,
    }
)

out_path = "/home/claude/churn_prediction/data/customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["churn"].value_counts(normalize=True))
