import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost

# --- Configuration ---
MODEL_PATH = 'Model_Fraud_Detection.pkl'
SCALER_PATH = 'Sclar.pkl'
ENCODER_PATH = 'Encoder.pkl'

# --- Load Model and Preprocessors ---
# Use st.cache_resource to load these objects only once
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and encoder."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Please ensure they are in the correct directory.")
        return None, None, None

model, scaler, encoder = load_assets()

# --- Preprocessing Function ---
def preprocess_input(data, scaler, encoder):
    # Separate categorical and numerical features
    categorical_cols = ['type']
    numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # Scale numerical features
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    # One-hot encode categorical features
    encoded_data = encoder.transform(data[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=data.index)

    # Combine scaled numerical and encoded categorical features
    processed_data = pd.concat([data[numerical_cols], encoded_df], axis=1)

    return processed_data

# --- Streamlit Web App Interface ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Real-Time Fraud Detection System")
st.markdown("Enter the transaction details below to check if it's fraudulent.")

# --- User Input Sidebar ---
st.sidebar.header("Transaction Details")

# Transaction type dropdown
type_options = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
transaction_type = st.sidebar.selectbox("Type of Transaction", type_options)

# Numerical inputs
amount = st.sidebar.number_input("Amount", min_value=0.0, format="%.2f")
step = st.sidebar.number_input("Step (Time)", min_value=1, step=1)
oldbalanceOrg = st.sidebar.number_input("Sender's Old Balance", min_value=0.0, format="%.2f")
newbalanceOrig = st.sidebar.number_input("Sender's New Balance", min_value=0.0, format="%.2f")
oldbalanceDest = st.sidebar.number_input("Recipient's Old Balance", min_value=0.0, format="%.2f")
newbalanceDest = st.sidebar.number_input("Recipient's New Balance", min_value=0.0, format="%.2f")

# --- Prediction Logic ---
if st.sidebar.button("Check Transaction"):
    if model and scaler and encoder:
        # Create a DataFrame from the user's input
        input_data = pd.DataFrame({
            'step': [step],
            'type': [transaction_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest]
        })

        with st.spinner("Analyzing transaction..."):
            # Preprocess the input data
            preprocessed_data = preprocess_input(input_data, scaler, encoder)

            # Make a prediction
            prediction = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)

            st.subheader("Analysis Result")
            if prediction[0] == 1:
                st.error("🚨 **FRAUD DETECTED** 🚨")
                st.write(f"Confidence Score: **{prediction_proba[0][1]*100:.2f}%**")
                st.warning("This transaction is highly suspicious. It is recommended to block it immediately and investigate.")
            else:
                st.success("✅ **Transaction Appears Legitimate** ✅")
                st.write(f"Confidence Score: **{prediction_proba[0][0]*100:.2f}%**")

    else:
        st.error("Model assets are not loaded. The application cannot make predictions.")

st.markdown("---")
st.info("This is a demo application. The model was trained on a public dataset and should not be used for real financial decisions.")

