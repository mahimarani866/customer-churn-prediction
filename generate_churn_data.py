import numpy as np
import pandas as pd

# Fix the random seed so that data looks the same every time we run it
np.random.seed(42)


def generate_churn_data(n_customers: int = 5000) -> pd.DataFrame:
    """
    Create a simple, realistic telecom customer churn dataset.
    """

    # Customer IDs like CUST_00001, CUST_00002, ...
    customer_id = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]

    # Basic info
    gender = np.random.choice(["Male", "Female"], size=n_customers)
    senior_citizen = np.random.choice([0, 1], size=n_customers, p=[0.8, 0.2])
    partner = np.random.choice(["Yes", "No"], size=n_customers, p=[0.45, 0.55])
    dependents = np.random.choice(["Yes", "No"], size=n_customers, p=[0.3, 0.7])

    # Tenure in months
    tenure = np.random.randint(0, 73, size=n_customers)

    # Phone & Internet service
    phone_service = np.random.choice(["Yes", "No"], size=n_customers, p=[0.9, 0.1])

    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_customers,
        p=[0.35, 0.5, 0.15],
    )

    def internet_option(service: str) -> str:
        if service == "No":
            return "No internet service"
        return np.random.choice(["Yes", "No"], p=[0.4, 0.6])

    online_security = [internet_option(s) for s in internet_service]
    online_backup = [internet_option(s) for s in internet_service]
    device_protection = [internet_option(s) for s in internet_service]
    tech_support = [internet_option(s) for s in internet_service]

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers,
        p=[0.6, 0.25, 0.15],
    )

    paperless_billing = np.random.choice(["Yes", "No"], size=n_customers, p=[0.7, 0.3])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_customers,
        p=[0.4, 0.2, 0.2, 0.2],
    )

    # Charges
    base_charge = 20
    internet_charge = np.where(
        internet_service == "No",
        0,
        np.where(internet_service == "DSL", 25, 45),
    )
    phone_charge = np.where(phone_service == "Yes", 15, 0)

    extras = []
    for os, ob, dp, ts in zip(online_security, online_backup, device_protection, tech_support):
        extra = 0
        if os == "Yes":
            extra += 5
        if ob == "Yes":
            extra += 5
        if dp == "Yes":
            extra += 4
        if ts == "Yes":
            extra += 7
        extras.append(extra)

    extras = np.array(extras)

    monthly_charges = base_charge + internet_charge + phone_charge + extras
    monthly_charges = monthly_charges + np.random.normal(0, 3, size=n_customers)
    monthly_charges = np.round(np.clip(monthly_charges, 15, None), 2)

    total_charges = monthly_charges * tenure + np.random.normal(0, 20, size=n_customers)
    total_charges = np.round(np.clip(total_charges, 0, None), 2)

    # Churn probability based on some simple rules
    churn_prob = np.full(n_customers, 0.15, dtype=float)
    churn_prob += np.where(contract == "Month-to-month", 0.15, 0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.05, 0)
    churn_prob += np.where(monthly_charges > 90, 0.08, 0)
    churn_prob += np.where(tenure < 6, 0.1, 0)
    churn_prob -= np.where(tenure > 24, 0.06, 0)
    churn_prob = np.clip(churn_prob, 0.01, 0.8)

    random_vals = np.random.rand(n_customers)
    churn = np.where(random_vals < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    return df


if __name__ == "__main__":
    # Generate the data
    df_customers = generate_churn_data(n_customers=5000)

    # Save into the data folder as churn_raw.csv
    output_path = "data/churn_raw.csv"
    df_customers.to_csv(output_path, index=False)

    print(f"✅ Saved dataset to {output_path}")
    print("First 5 rows:")
    print(df_customers.head())
    print("\nChurn rate (Yes/No):")
    print(df_customers['Churn'].value_counts(normalize=True))
