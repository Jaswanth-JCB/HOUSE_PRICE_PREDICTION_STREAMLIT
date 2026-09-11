import streamlit as st
import pandas as pd
import joblib

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# Load Trained Model
# =========================================================
model = joblib.load("house_price_model.pkl")

# =========================================================
# Application Title
# =========================================================
st.title("🏠 House Price Prediction")
st.write(
    "Enter the house and area information "
    "to predict the median house value."
)

# =========================================================
# User Input
# =========================================================
st.subheader("Enter House Information")

col1, col2, col3 = st.columns(3)

with col1:
    CRIM = st.number_input("CRIM", min_value=0.0, value=1.0)
    ZN = st.number_input("ZN", min_value=0.0, value=0.0)
    INDUS = st.number_input("INDUS", min_value=0.0, value=10.0)
    CHAS = st.number_input("CHAS", min_value=0.0, max_value=1.0, value=0.0)
    NOX = st.number_input("NOX", min_value=0.0, value=0.5)

with col2:
    RM = st.number_input("RM", min_value=0.0, value=6.0)
    AGE = st.number_input("AGE", min_value=0.0, max_value=100.0, value=50.0)
    DIS = st.number_input("DIS", min_value=0.0, value=4.0)
    RAD = st.number_input("RAD", min_value=0.0, value=5.0)

with col3:
    TAX = st.number_input("TAX", min_value=0.0, value=300.0)
    PTRATIO = st.number_input("PTRATIO", min_value=0.0, value=18.0)
    B = st.number_input("B", min_value=0.0, value=350.0)
    LSTAT = st.number_input("LSTAT", min_value=0.0, value=12.0)

# =========================================================
# Prediction Button
# =========================================================
st.write("")

if st.button("🏠 Predict House Price", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "CRIM": [CRIM],
        "ZN": [ZN],
        "INDUS": [INDUS],
        "CHAS": [CHAS],
        "NOX": [NOX],
        "RM": [RM],
        "AGE": [AGE],
        "DIS": [DIS],
        "RAD": [RAD],
        "TAX": [TAX],
        "PTRATIO": [PTRATIO],
        "B": [B],
        "LSTAT": [LSTAT]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(f"Predicted House Value: {prediction:.2f}")