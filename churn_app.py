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

# Streamlit Config
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
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0)
    st.success("Data cleaned!")

    # 3. Feature Engineering
    st.header("3. Feature Engineering")
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                               labels=['0-12', '12-24', '24-48', '48-72'])
    df = pd.get_dummies(df, drop_first=True)
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
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # 5. Train/Test Split
    st.header("5. Model Training & Evaluation")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Random Forest
    st.subheader("Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    st.text("Classification Report:\n" + classification_report(y_test, rf_pred))
    st.text(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.2f}")

    # XGBoost
    st.subheader("XGBoost")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
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

    # 6. ROC Curve
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

    # 7. SHAP Analysis
    st.header("6. Model Explainability (SHAP)")
    shap.initjs()
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test)

    st.subheader("Top Features by SHAP Value")
    fig_shap = shap.plots.bar(shap_values, show=False)
    st.pyplot(bbox_inches='tight')
