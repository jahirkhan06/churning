import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle

# Title
st.title("Telco Customer Churn Prediction App")

# Load Model (Assuming you saved the XGBoost model)
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")  # Save using model.save_model('xgb_model.json')

# Input fields
st.header("Customer Information")

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72)
MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

# You can expand to include more features if needed

# Manual feature conversion
data = {
    'gender_Male': 1 if gender == 'Male' else 0,
    'SeniorCitizen': SeniorCitizen,
    'Partner_Yes': 1 if Partner == 'Yes' else 0,
    'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

# Placeholder for other required features
# You must align this dictionary with the features used in training

input_df = pd.DataFrame([data])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Prediction: {'Will Churn' if prediction[0] == 1 else 'Will Not Churn'}")
    st.write(f"Churn Probability: {prob:.2f}")
    xgb_model.save_model("xgb_model.json")
