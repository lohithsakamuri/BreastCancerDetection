import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load model + scaler
model = joblib.load("breast_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

st.title("Breast Cancer Detection Web App")
st.write("Enter tumor feature values and click Predict.")

inputs = []

# Take 30 inputs
for name in feature_names:
    val = st.number_input(name, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    arr = np.array(inputs).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]

    if pred == 0:
        st.error("Prediction: Malignant (Cancer)")
    else:
        st.success("Prediction: Benign (Not Cancer)")
