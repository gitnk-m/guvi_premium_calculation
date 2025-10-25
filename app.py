import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import date

# Load your pre-trained model
@st.cache_resource
def load_model():
    with open("Premium_xgb_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.set_page_config(page_title="Smart Premium", layout="wide")
st.title("🏦 Smart Premium App")
st.markdown("Fill the details below to get the predicted premium value or risk score.")

# --- User Inputs ---
with st.form("input_form"):
    st.header("📋 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        annual_income = st.number_input("Annual Income (in ₹)", min_value=0)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Number of Dependents", min_value=0, step=1)
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    with col2:
        occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
        health_score = st.slider("Health Score", 0, 100, 70)
        location = st.selectbox("Location",["Urban", "Suburban", "Rural"])
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
        previous_claims = st.number_input("Previous Claims", min_value=0, step=1)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, step=1)

    with col3:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        insurance_duration = st.number_input("Insurance Duration (in years)", min_value=0.0, step=0.5)
        policy_start_date = st.date_input("Policy Start Date", date.today())
        feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
        smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
        exercise_freq = st.selectbox("Exercise Frequency", ["Rarely", "Weekly", "Daily", "Monthly"])
        property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    # Derived features
    policy_start_year = policy_start_date.year
    policy_start_month = policy_start_date.month
    policy_start_day = policy_start_date.day

    submitted = st.form_submit_button("🔍 Predict")

gender_map = {"Male": 1, "Female": 0}
marital_status_map = {"Single": 2, "Married": 1, "Divorced": 0}
education_map = {"High School": 1, "Bachelor's": 0, "Master's": 2, "PhD": 3}
occupation_map = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
location_map = {"Urban": 2, "Suburban": 1, "Rural": 0}
policy_type_map = {"Basic": 0, "Comprehensive": 1, "Premium": 2}
feedback_map = {"Good": 2, "Average": 1, "Poor": 0}
smoking_status_map = {"Yes": 1, "No": 0}
exercise_freq_map = {"Rarely": 2, "Weekly": 2, "Daily": 0, "Monthly": 1}
property_type_map = {"House": 2, "Apartment": 0, "Condo": 1}

# Map categorical inputs to numerical values
if submitted:
    # --- Prepare input for model ---
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_map[gender],
        "Annual Income": annual_income,
        "Marital Status": marital_status_map[marital_status],
        "Number of Dependents": dependents,
        "Education Level": education_map[education] ,
        "Occupation": occupation_map[occupation],
        "Health Score": health_score,
        "Location": location_map[location],
        "Policy Type": policy_type_map[policy_type],
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        # "Policy Start Date": policy_start_date,
        "Customer Feedback": feedback_map[feedback],
        "Smoking Status": smoking_status_map[smoking_status],
        "Exercise Frequency": exercise_freq_map[exercise_freq],
        "Property Type": property_type_map[property_type],
        "Policy_start_year": policy_start_year,
        "Policy_start_month": policy_start_month,
        "Policy_start_day": policy_start_day
    }])

    # Handle categorical encoding if your model expects it
    # (Assumes your model pipeline already includes preprocessing)
    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Value: **₹{prediction[0]:,.2f}**")
