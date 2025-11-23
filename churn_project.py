"""
Customer Churn Prediction & Revenue Risk Analysis
=================================================

This file is a plain-English summary of the whole project.

It explains:
1) How I generated the dataset in generate_churn_data.py
2) How I trained and evaluated the model in train_churn_model.py
3) How to re-run the full pipeline from code.

--------------------------------------------------
1. DATA GENERATION: generate_churn_data.py
--------------------------------------------------

File: generate_churn_data.py

Main idea:
    I do NOT download any external dataset.
    Instead, I create a synthetic telecom-style dataset using numpy + pandas.

What the script does:
    - Creates ~5,000 customers with IDs like CUST_00001, CUST_00002, ...
    - For each customer, it randomly (but realistically) assigns:
        * gender (Male / Female)
        * SeniorCitizen (0/1)
        * Partner (Yes/No)
        * Dependents (Yes/No)
        * tenure in months (0–72)
        * PhoneService (Yes/No)
        * MultipleLines (Yes/No/No phone service)
        * InternetService (DSL / Fiber optic / No)
        * OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
        * Contract (Month-to-month / One year / Two year)
        * PaperlessBilling (Yes/No)
        * PaymentMethod (Electronic check, Mailed check, Bank transfer, Credit card)
        * MonthlyCharges (base + internet + phone + add-ons)
        * TotalCharges (MonthlyCharges * tenure with noise)

Business logic used to create the "Churn" label:
    - I start from a base churn probability (around 15%).
    - Then I adjust the probability based on business assumptions:
        * Month-to-month contract -> more likely to churn
        * Fiber optic and very high monthly charges -> more likely to churn
        * Very new customers (tenure < 6 months) -> more likely to churn
        * Long-tenure customers (tenure > 24 months) -> less likely to churn
    - I clip the probability between 1% and 80%.
    - Finally, I draw a random number for each customer and decide:
        * If random < churn_prob -> Churn = "Yes"
        * Else -> Churn = "No"

Output:
    - The script saves everything to data/churn_raw.csv.
    - This CSV is the starting point for the machine learning pipeline.

--------------------------------------------------
2. MODELLING & SCORING: train_churn_model.py
--------------------------------------------------

File: train_churn_model.py

Goal:
    Train a classification model to predict whether a customer will churn,
    and calculate a churn probability and risk level for each customer.

Steps inside this script:

  1) Load data
     - Read data/churn_raw.csv into a pandas DataFrame.
     - Use all columns except "customerID" and "Churn" as features (X).
     - Convert target "Churn" into numeric y (Yes -> 1, No -> 0).

  2) Feature preprocessing
     - Automatically detect numeric columns:
         * tenure, SeniorCitizen, MonthlyCharges, TotalCharges, etc.
     - Automatically detect categorical columns:
         * gender, Partner, Dependents, InternetService, Contract, etc.
     - Build a ColumnTransformer:
         * pass numeric features through unchanged
         * apply OneHotEncoder(handle_unknown="ignore") to categorical features

  3) Model
     - Use LogisticRegression(max_iter=1000) inside a sklearn Pipeline.
     - The Pipeline = [preprocess] -> [classifier].

  4) Train/Test split
     - Split the dataset into 80% train, 20% test with stratify=y
       so the churn ratio stays similar in both sets.

  5) Training and evaluation
     - Fit the Pipeline on the train data.
     - Evaluate on the test data:
         * Accuracy
         * ROC AUC
         * Classification report (precision, recall, f1-score)

  6) Feature importance (explainability)
     - Refit the model on the full dataset (X, y).
     - Extract:
         * the trained LogisticRegression model
         * the fitted OneHotEncoder feature names
     - Combine them into a single list of features.
     - Take the absolute value of the logistic regression coefficients
       as a measure of importance.
     - Sort features by importance and save to:
         data/churn_feature_importance.csv

  7) Scoring and business impact
     - Use the final model to predict churn probability for *every* customer.
     - Add these new columns:
         * Churn_Probability  (float 0–1)
         * Churn_Predicted    ("Yes"/"No")
         * Churn_Risk_Level   ("Low"/"Medium"/"High") based on probability

     - Estimate revenue risk:
         * Assume we care about the next 12 months.
         * Expected_Annual_Revenue  = MonthlyCharges * 12
         * Expected_Revenue_Loss    = Churn_Probability * Expected_Annual_Revenue

     - Sort customers so that high risk + high revenue customers appear first.
     - Save everything to:
         data/churn_scored.csv

--------------------------------------------------
3. HOW TO RUN THE FULL PIPELINE FROM PYTHON
--------------------------------------------------

From a terminal in this project folder, you can do:

    python churn_project.py

This will:
    1) Regenerate the synthetic dataset (data/churn_raw.csv)
    2) Retrain the model and rescore all customers
    3) Update:
         - data/churn_raw.csv
         - data/churn_scored.csv
         - data/churn_feature_importance.csv

The run_pipeline() function below is just a simple wrapper around the two scripts.
"""

from generate_churn_data import generate_churn_data
from train_churn_model import main as train_model


def run_pipeline():
    """
    Convenience wrapper to run the full project in one go.

    1) Generate synthetic data and save to data/churn_raw.csv
    2) Train the model, compute feature importance, and score all customers
    """
    print("📊 Customer Churn Prediction & Revenue Risk Analysis")
    print("-" * 70)

    # Step 1: Generate data
    print("Step 1/2: Generating synthetic customer data...")
    df_customers = generate_churn_data(n_customers=5000)
    df_customers.to_csv("data/churn_raw.csv", index=False)
    print("   ✅ Saved dataset to data/churn_raw.csv")

    # Step 2: Train model and score customers
    print("\nStep 2/2: Training model and scoring customers...")
    train_model()
    print("\n✅ Pipeline finished. Check the data/ folder for CSV outputs.")
    print("   - churn_raw.csv")
    print("   - churn_scored.csv")
    print("   - churn_feature_importance.csv")


if __name__ == "__main__":
    run_pipeline()
