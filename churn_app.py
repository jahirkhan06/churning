
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Telco Customer Churn - Full ML Pipeline")

# 1. Upload Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df.head())

    # 2. Data Cleaning
    st.header("2. Data Cleaning")
    df.drop('customerID', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    st.success("Data cleaned!")

    # 3. Feature Engineering
    st.header("3. Feature Engineering")
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72'])

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('Churn')
    df = pd.get_dummies(df, columns=cat_cols + ['TenureGroup'], drop_first=True)

    st.write("Processed Data", df.head())

    # 4. Exploratory Data Analysis
    st.header("4. Exploratory Data Analysis")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # 5. Model Training & Evaluation
    st.header("5. Model Training & Evaluation")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Random Forest
    st.subheader("Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    st.text("Classification Report:\n" + classification_report(y_test, rf_pred))
    st.text(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.2f}")

    # XGBoost
    st.subheader("XGBoost")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    st.text("Classification Report:\n" + classification_report(y_test, xgb_pred))
    st.text(f"ROC-AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix (XGBoost)")
    cm = confusion_matrix(y_test, xgb_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Feature Importances
    st.subheader("Feature Importances (XGBoost)")
    importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    st.dataframe(importance_df.sort_values(by="Importance", ascending=False).head(10))

    # ROC Curve
    st.subheader("ROC Curve Comparison")
    rf_probs = rf.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_rf, tpr_rf, label='Random Forest')
    ax_roc.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    st.pyplot(fig_roc)

    # SHAP Explainability
    st.header("6. Model Explainability (SHAP)")
    try:
        shap.initjs()
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test.sample(100, random_state=42))
        st.subheader("Top Features by SHAP Value")
        shap.plots.bar(shap_values, show=False)
        fig_shap = plt.gcf()
        st.pyplot(fig_shap)
    except Exception as e:
        st.warning(f"SHAP plot skipped due to: {e}")

    # Manual Prediction
    st.header("7. Manual Churn Prediction")
    with st.form("prediction_form"):
        st.subheader("Enter Customer Info:")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
        total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)
        tenure_group = pd.cut([tenure], bins=[0, 12, 24, 48, 72],
                              labels=['0-12', '12-24', '24-48', '48-72'])[0]

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = {
                'SeniorCitizen': senior,
                'tenure': tenure,
                'MonthlyCharges': monthly,
                'TotalCharges': total,
                f'gender_{gender}': 1,
                f'Partner_{partner}': 1,
                f'Dependents_{dependents}': 1,
                f'TenureGroup_{tenure_group}': 1
            }

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            prediction = xgb_model.predict(input_df)[0]
            prob = xgb_model.predict_proba(input_df)[0][1]

            st.success(f"Prediction: {'Will Churn' if prediction == 1 else 'Will Not Churn'}")
            st.info(f"Probability of Churn: {prob:.2f}")
