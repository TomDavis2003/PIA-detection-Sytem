import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from data_processing import preprocess_new_prompt
from shap_analysis import shap_analysis_lstm
import warnings
import tensorflow as tf

# Suppress TensorFlow and SHAP warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load models and preprocessors
try:
    lstm_model = load_model("lstm_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    max_len = joblib.load("max_len.pkl")
except Exception as e:
    st.error(f"Failed to load models or preprocessors: {str(e)}")
    st.stop()

# Recompile LSTM model
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Streamlit configuration
st.set_page_config(page_title="Prompt Injection Detector", layout="wide")
st.title("Prompt Injection Attack Detector")

tab1, tab2, tab3 = st.tabs(["Prompt Classification", "Dataset Overview", "About Us"])

with tab1:
    st.header("Detect Malicious Prompts")
    user_input = st.text_area("Enter your prompt here:", height=150)

    if st.button("Analyze"):
        if not user_input:
            st.warning("Please enter a prompt to analyze.")
        else:
            try:
                # Preprocess the prompt
                padded_sequence = preprocess_new_prompt(user_input, tokenizer, max_len)
                st.write(f"Padded Sequence Shape: {padded_sequence.shape}")

                # LSTM Prediction
                lstm_pred = lstm_model.predict(padded_sequence, verbose=0)[0]
                lstm_pred_percent = {label: prob * 100 for label, prob in zip(label_encoder.classes_, lstm_pred)}

                # Display Results
                st.subheader("Prediction Results")
                st.write("🔹 **LSTM Predictions:**")
                for label, percent in lstm_pred_percent.items():
                    st.write(f"{label}: {percent:.2f}%")

                # SHAP Analysis
                st.subheader("SHAP Analysis")
                st.write("**LSTM SHAP Summary:**")
                lstm_importance = shap_analysis_lstm(lstm_model, tokenizer, user_input, padded_sequence, max_len)
                if lstm_importance:
                    st.write("Top 5 Tokens Influencing LSTM Prediction:")
                    for token, value in lstm_importance:
                        st.write(f"{token}: {value:.4f}")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

with tab2:
    st.header("Dataset Overview")
    try:
        df = pd.read_csv("PIA_Augmented_Dataset.csv")
        st.dataframe(df.head())
        st.bar_chart(df["Category"].value_counts())
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")

with tab3:
    st.header("About")
    st.write("This project detects **Prompt Injection Attacks** using LSTM and Explainable AI (XAI).")
    st.write("Developed by Akshara Balan, Alan C M, Maria Baiju and Tom Davis under the guidance of Dr. Amrutha Muralidharan Nair")
