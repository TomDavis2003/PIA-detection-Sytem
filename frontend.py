import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from data_processing import preprocess_new_prompt
from shap_analysis import shap_analysis_lstm
import warnings
import tensorflow as tf
import os

# Environment & warning suppression
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load models and preprocessors with validation
try:
    lstm_model = load_model("lstm_model.keras")

    tokenizer = joblib.load("tokenizer.pkl")
    if tokenizer is None:
        raise ValueError("Loaded tokenizer is None")

    label_encoder = joblib.load("label_encoder.pkl")
    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        raise ValueError("Loaded label encoder is invalid or None")

    max_len = joblib.load("max_len.pkl")
    if max_len is None or not isinstance(max_len, int):
        raise ValueError("max_len is None or invalid")

except Exception as e:
    st.error(f"❌ Failed to load models or preprocessors: {str(e)}")
    st.stop()

# Recompile model (necessary after loading)
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Streamlit UI setup
st.set_page_config(page_title="Prompt Injection Detector", layout="wide")
st.title("🛡️ Prompt Injection Attack Detector")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prompt Classification", "Dataset Overview", "About Us"])

# ──────────────── TAB 1 ────────────────
with tab1:
    st.header("🔍 Detect Malicious Prompts")
    user_input = st.text_area("Enter your prompt here:", height=150)

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a prompt to analyze.")
        else:
            try:
                # Preprocess input
                padded_sequence = preprocess_new_prompt(user_input, tokenizer, max_len)
                if padded_sequence is None:
                    raise ValueError("Preprocessing returned None.")

                st.write(f"Padded Sequence Shape: `{padded_sequence.shape}`")

                # Model prediction
                lstm_pred = lstm_model.predict(padded_sequence, verbose=0)[0]
                lstm_pred_percent = {
                    label: prob * 100 for label, prob in zip(label_encoder.classes_, lstm_pred)
                }

                # Output prediction
                st.subheader("📊 Prediction Results")
                st.write("🔹 **LSTM Predictions:**")
                for label, percent in lstm_pred_percent.items():
                    st.write(f"- {label}: **{percent:.2f}%**")

                # SHAP Explainability
                st.subheader("🧠 SHAP Analysis")
                st.write("**Top Tokens Influencing Prediction:**")
                lstm_importance = shap_analysis_lstm(lstm_model, tokenizer, user_input, padded_sequence, max_len)
                if lstm_importance:
                    for token, value in lstm_importance[:5]:
                        st.write(f"- `{token}`: {value:.4f}")
                else:
                    st.info("No SHAP importance scores available.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# ──────────────── TAB 2 ────────────────
with tab2:
    st.header("📁 Dataset Overview")
    try:
        df = pd.read_csv("PIA_Augmented_Dataset.csv")
        st.dataframe(df.head())
        st.bar_chart(df["Category"].value_counts())
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")

# ──────────────── TAB 3 ────────────────
with tab3:
    st.header("👩‍💻 About")
    st.write("""
    This project detects **Prompt Injection Attacks** using LSTM and Explainable AI (XAI).
    
    **Developed by**:  
    Akshara Balan, Alan C M, Maria Baiju, Tom Davis  
    **Guided by**: Dr. Amrutha Muralidharan Nair
    """)
