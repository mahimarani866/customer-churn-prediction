import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def risk_bucket(p: float) -> str:
    """
    Convert churn probability into a simple risk label.
    """
    if p >= 0.7:
        return "High"
    elif p >= 0.4:
        return "Medium"
    else:
        return "Low"


def main():
    # 1. Load the dataset we created earlier
    df = pd.read_csv("data/churn_raw.csv")
    print("Loaded data with shape:", df.shape)
    print("Columns:", list(df.columns))

    # 2. Separate features (X) and target (y)
    target_col = "Churn"

    # We do NOT use customerID as a feature
    X = df.drop(columns=[target_col, "customerID"])
    y = df[target_col].map({"No": 0, "Yes": 1})  # convert Yes/No to 1/0

    # 3. Find numeric and categorical columns automatically
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 4. Preprocessing:
    #    - numeric: keep as is
    #    - categorical: One-Hot Encode
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 5. Define the model (Logistic Regression)
    log_reg = LogisticRegression(max_iter=1000)

    # 6. Create the full pipeline: preprocessing + model
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", log_reg),
        ]
    )

    # 7. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

    # 8. Fit the model on training data
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # 9. Evaluate on test data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n===== Model Performance =====")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # 10. Refit model on FULL data (for scoring + explainability)
    print("\nRefitting model on full dataset for scoring and feature importance...")
    model.fit(X, y)

    # ------------------------------------------------------------------
    # A) FEATURE IMPORTANCE (what drives churn?)
    # ------------------------------------------------------------------
    # Get trained classifier and preprocessor from pipeline
    clf = model.named_steps["classifier"]
    fitted_preprocessor = model.named_steps["preprocess"]

    # Numeric feature names
    numeric_cols = fitted_preprocessor.transformers_[0][2]

    # OneHotEncoder and its feature names
    ohe = fitted_preprocessor.named_transformers_["cat"]
    ohe_input_cols = fitted_preprocessor.transformers_[1][2]
    ohe_feature_names = ohe.get_feature_names_out(ohe_input_cols)

    # Combine all feature names
    all_feature_names = list(numeric_cols) + list(ohe_feature_names)

    # Coefficients from logistic regression (importance ~ |coef|)
    coefs = clf.coef_[0]
    importance_df = pd.DataFrame(
        {"feature": all_feature_names, "importance": np.abs(coefs)}
    ).sort_values("importance", ascending=False)

    importance_path = "data/churn_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nTop 10 most important features driving churn:")
    print(importance_df.head(10))
    print(f"\n✅ Saved feature importance report to {importance_path}")

    # ------------------------------------------------------------------
    # B) SCORE EVERY CUSTOMER + BUSINESS IMPACT
    # ------------------------------------------------------------------
    churn_proba_all = model.predict_proba(X)[:, 1]
    churn_pred_all = model.predict(X)

    df_scored = df.copy()
    df_scored["Churn_Probability"] = churn_proba_all
    df_scored["Churn_Predicted"] = np.where(churn_pred_all == 1, "Yes", "No")
    df_scored["Churn_Risk_Level"] = df_scored["Churn_Probability"].apply(risk_bucket)

    # Estimate revenue impact: assume we care about next 12 months
    revenue_horizon_months = 12
    df_scored["Expected_Annual_Revenue"] = df_scored["MonthlyCharges"] * revenue_horizon_months
    df_scored["Expected_Revenue_Loss"] = (
        df_scored["Churn_Probability"] * df_scored["Expected_Annual_Revenue"]
    )

    total_expected_loss = df_scored["Expected_Revenue_Loss"].sum()
    print(f"\nEstimated TOTAL expected revenue at risk: {total_expected_loss:,.2f}")

    # Sort so that highest-risk, highest-value customers come first
    df_scored = df_scored.sort_values(
        by=["Churn_Risk_Level", "Expected_Revenue_Loss"],
        ascending=[False, False],
    )

    # 12. Save to a new CSV file
    output_path = "data/churn_scored.csv"
    df_scored.to_csv(output_path, index=False)

    print(f"\n✅ Saved scored data (with revenue impact) to {output_path}")
    print("Sample rows:")
    print(df_scored.head())


if __name__ == "__main__":
    main()
