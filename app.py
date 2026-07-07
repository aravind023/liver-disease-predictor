import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Liver Disease Predictor", page_icon="\U0001FA7A", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("liver_disease_model.pkl")
    scaler = joblib.load("liver_disease_scaler.pkl")
    features = joblib.load("liver_disease_features.pkl")
    return model, scaler, features

model, scaler, feature_order = load_artifacts()

st.title("Liver Disease Prediction")
st.write(
    "Enter a patient's lab values below to predict the likelihood of liver disease. "
    "This tool is for educational purposes and is **not** a substitute for medical advice."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
        alkphos = st.number_input("Alkaline Phosphotase", min_value=0.0, value=200.0, step=1.0)

    with col2:
        sgpt = st.number_input("Alamine Aminotransferase (SGPT)", min_value=0.0, value=30.0, step=1.0)
        sgot = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=0.0, value=35.0, step=1.0)
        total_protein = st.number_input("Total Proteins", min_value=0.0, value=6.5, step=0.1)
        albumin = st.number_input("Albumin", min_value=0.0, value=3.2, step=0.1)
        ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.05)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alkphos,
        "Alamine_Aminotransferase": sgpt,
        "Aspartate_Aminotransferase": sgot,
        "Total_Protiens": total_protein,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": ag_ratio,
    }
    input_df = pd.DataFrame([input_dict])[feature_order]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"Prediction: **Liver Disease Likely** (probability: {probability:.1%})")
    else:
        st.success(f"Prediction: **No Liver Disease Indicated** (probability of disease: {probability:.1%})")

    st.caption("Model: trained on the Indian Liver Patient Dataset (ILPD). See the accompanying notebook for full methodology.")
