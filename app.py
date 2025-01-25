import numpy as np
from pandas import options
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("DIABetes_model (6).pkl")


st.cache_resource.clear()

st.title('Diabetes Prediction app')
st.write(
    """This app aims to deliver practical insights for health measures and enhance early diagnosis efforts.

            **Why is this important?**

            The increasing prevalence of diabetes poses significant challenges to public health systems. Understanding its impact is crucial for developing effective interventions, promoting early diagnosis, and encouraging preventive measures to mitigate its effects on individuals and communities.""")


model=load_model()

if model:
    st.header("Enter Patient Information")

heart_disease = st.selectbox("Do you have heart disease?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
heart_disease_value = heart_disease[0]  # Extract the numeric value

hypertension = st.selectbox("Do you have hypertension (high blood pressure)", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
hypertension_value = hypertension[0]  # Extract the numeric value

age = st.number_input("Enter your age", min_value=0, max_value=100, value=0)
bmi = st.number_input("Enter your BMI(Body Mass Index)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
blood_glucose_level = st.number_input("Enter your Blood Glucose Level(mg/dL)", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
HbA1c_level = st.number_input("Enter your HbA1c Level (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Use only numeric values
input_data = np.array([age, hypertension_value, heart_disease_value, bmi, HbA1c_level, blood_glucose_level])

# Pass a unique key to the button
if st.button("Asses Diabetes Risk", key="predict_button"):
    prediction = model.predict(input_data.reshape(1, -1))  # Ensure the input is in the correct shape
    diabetes_status = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"Predicted Diabetes Risk: {diabetes_status}")

st.write("""**Disclaimer:** This app is for informational purposes only and should not be considered medical advice. 
Consult with a healthcare professional for any health concerns""")