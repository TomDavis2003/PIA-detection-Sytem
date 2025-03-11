# frontend.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_processing import preprocess_new_prompt
from shap_analysis import shap_analysis_rf, shap_analysis_lstm, generate_full_shap_analysis

def run_frontend(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len):
    st.set_page_config(page_title="Prompt Injection Detection System", layout="wide")
    
    # Streamlit app title and description
    st.title("Prompt Injection Attack Detection System")
    st.write("Enter a prompt below to classify it as Legitimate or Malicious using Random Forest and LSTM models with SHAP explanations.")
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analysis Tool", "Global SHAP Analysis", "About"])
    
    if page == "Analysis Tool":
        run_analysis_tool(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len)
    elif page == "Global SHAP Analysis":
        run_global_shap_analysis(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len)
    else:
        run_about_page()

def run_analysis_tool(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len):
    # Input field for the prompt
    prompt = st.text_area("Enter your prompt here:", height=100)
    
    # Button to trigger prediction
    if st.button("Analyze Prompt"):
        if not prompt:
            st.warning("Please enter a prompt to analyze.")
        else:
            # Show a spinner while processing
            with st.spinner("Analyzing prompt..."):
                # Preprocess new prompt
                tfidf_features, padded_sequence = preprocess_new_prompt(prompt, tfidf_vectorizer, tokenizer, max_len)
                
                # Create two columns for the models
                col1, col2 = st.columns(2)
                
                # Random Forest Prediction
                rf_probs = rf_model.predict_proba(tfidf_features)[0]
                rf_pred_idx = rf_probs.argmax()
                rf_pred = label_encoder.inverse_transform([rf_pred_idx])[0]
                
                # Find indices for legitimate and malicious classes
                legitimate_idx = -1
                malicious_idx = -1
                for i, cls in enumerate(label_encoder.classes_):
                    if "legitimate" in cls.lower():
                        legitimate_idx = i
                    elif "malicious" in cls.lower() or "injection" in cls.lower():
                        malicious_idx = i
                
                # Calculate probabilities
                rf_legit_prob = rf_probs[legitimate_idx] * 100 if legitimate_idx >= 0 else 0
                rf_mal_prob = rf_probs[malicious_idx] * 100 if malicious_idx >= 0 else 0
                
                # LSTM Prediction
                lstm_probs = lstm_model.predict(padded_sequence, verbose=0)[0]
                lstm_pred_idx = lstm_probs.argmax()
                lstm_pred = label_encoder.inverse_transform([lstm_pred_idx])[0]
                
                # Calculate probabilities
                lstm_legit_prob = lstm_probs[legitimate_idx] * 100 if legitimate_idx >= 0 else 0
                lstm_mal_prob = lstm_probs[malicious_idx] * 100 if malicious_idx >= 0 else 0
                
                # Random Forest Results
                with col1:
                    st.subheader("Random Forest Prediction")
                    
                    # Prediction result with colored background
                    if "legitimate" in rf_pred.lower():
                        st.success(f"**Prediction:** {rf_pred}")
                    else:
                        st.error(f"**Prediction:** {rf_pred}")
                    
                    # Probability bar chart
                    probabilities = {
                        "Legitimate": rf_legit_prob,
                        "Malicious": rf_mal_prob
                    }
                    
                    # Display probabilities as a horizontal bar chart
                    chart_data = pd.DataFrame({
                        "Probability": [rf_legit_prob, rf_mal_prob],
                        "Class": ["Legitimate", "Malicious"]
                    })
                    
                    st.bar_chart(chart_data.set_index("Class"))
                    
                    # SHAP Analysis
                    st.subheader("Feature Importance (SHAP)")
                    rf_shap = shap_analysis_rf(rf_model, tfidf_vectorizer, prompt, tfidf_features)
                    
                    # Display SHAP values with color coding
                    for feature, value in rf_shap:
                        if value > 0:
                            st.markdown(f"**{feature}**: <span style='color:red'>+{value:.4f}</span> (increases likelihood of {rf_pred})", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{feature}**: <span style='color:blue'>{value:.4f}</span> (decreases likelihood of {rf_pred})", unsafe_allow_html=True)
                
                # LSTM Results
                with col2:
                    st.subheader("LSTM Prediction")
                    
                    # Prediction result with colored background
                    if "legitimate" in lstm_pred.lower():
                        st.success(f"**Prediction:** {lstm_pred}")
                    else:
                        st.error(f"**Prediction:** {lstm_pred}")
                    
                    # Probability bar chart
                    chart_data = pd.DataFrame({
                        "Probability": [lstm_legit_prob, lstm_mal_prob],
                        "Class": ["Legitimate", "Malicious"]
                    })
                    
                    st.bar_chart(chart_data.set_index("Class"))
                    
                    # SHAP Analysis
                    st.subheader("Token Importance (SHAP)")
                    lstm_shap = shap_analysis_lstm(lstm_model, tokenizer, prompt, padded_sequence, max_len)
                    
                    # Display SHAP values with color coding
                    for token, value in lstm_shap:
                        if value > 0:
                            st.markdown(f"**{token}**: <span style='color:red'>+{value:.4f}</span> (increases likelihood of {lstm_pred})", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{token}**: <span style='color:blue'>{value:.4f}</span> (decreases likelihood of {lstm_pred})", unsafe_allow_html=True)
    
    # Display explanation of SHAP values
    with st.expander("What do these explanations mean?"):
        st.write("""
        **SHAP (SHapley Additive exPlanations) values** show how each feature influences the model's prediction:
        
        - **Positive values (red)** indicate the feature pushes the prediction toward the predicted class
        - **Negative values (blue)** indicate the feature pushes the prediction away from the predicted class
        - **Larger absolute values** indicate a stronger influence on the prediction
        
        For example, if a word has a large positive SHAP value and the model predicted "Malicious", this word strongly contributed to classifying the prompt as malicious.
        """)

def run_global_shap_analysis(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len):
    st.header("Global SHAP Analysis")
    st.write("""
    This page shows global feature importance analysis that helps explain what features 
    the models look at when detecting prompt injection attacks.
    """)
    
    # Check if global analysis has been run before
    if not os.path.exists("./shap_plots/rf/global_importance.png"):
        if st.button("Generate Global SHAP Analysis"):
            with st.spinner("Generating global SHAP analysis... This may take a few minutes."):
                # We need to load a sample of the dataset for this analysis
                from data_processing import load_and_preprocess_data
                
                st.info("Loading and preprocessing data...")
                file_path = 'PIA_Augmented_Dataset.csv'  # This should match your dataset path
                
                try:
                    data = load_and_preprocess_data(file_path)
                    if data is None:
                        st.error("Failed to load data for SHAP analysis.")
                        return
                    
                    X_tfidf, X_padded, y, _, _, _, max_len = data
                    
                    # Generate the analysis
                    st.info("Generating SHAP analysis for both models...")
                    results = generate_full_shap_analysis(
                        rf_model, lstm_model, X_tfidf, X_padded, y, 
                        tfidf_vectorizer, tokenizer, label_encoder, max_len
                    )
                    
                    st.success("Global SHAP analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error during SHAP analysis: {str(e)}")
    else:
        # Display the pre-generated plots
        st.subheader("Random Forest Global Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.image("./shap_plots/rf/global_importance.png", caption="Global Feature Importance (SHAP Summary Plot)")
            except:
                st.error("Failed to load RF global importance plot.")
        
        with col2:
            try:
                st.image("./shap_plots/rf/top20_features.png", caption="Top 20 Features by Importance")
            except:
                st.error("Failed to load RF top 20 features plot.")
        
        st.subheader("LSTM Model Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            try:
                st.image("./shap_plots/lstm/position_importance.png", caption="Token Position Importance")
            except:
                st.error("Failed to load LSTM position importance plot.")
        
        with col4:
            try:
                st.image("./shap_plots/lstm/top20_tokens.png", caption="Top 20 Tokens by Importance")
            except:
                st.error("Failed to load LSTM top 20 tokens plot.")
        
        # Add explanation
        st.markdown("""
        ### Interpreting These Plots
        
        **Random Forest Analysis:**
        - The summary plot shows how each feature pushes the prediction toward (red) or away from (blue) the target.
        - The bar chart shows the top 20 features with the highest average impact on model predictions.
        
        **LSTM Analysis:**
        - The position importance plot shows which positions in the sequence have the most influence on predictions.
        - The top tokens chart shows which specific words or tokens have the highest impact across all samples.
        
        This analysis helps identify patterns that can indicate prompt injection attacks, such as specific keywords, 
        command structures, or patterns that attackers commonly use.
        """)

def run_about_page():
    st.header("About the Prompt Injection Attack Detection System")
    
    st.markdown("""
    ### Project Overview
    
    This system uses machine learning to detect and prevent prompt injection attacks, which are attempts to manipulate 
    large language models (LLMs) through carefully crafted inputs.
    
    ### Models Used
    
    1. **Random Forest**
    - Uses TF-IDF vectorization of text
    - Provides interpretable feature importance
    - Effective for detecting known attack patterns
    
    2. **LSTM (Long Short-Term Memory)**
    - Neural network that understands sequential data
    - Captures context and relationships between words
    - Better for detecting novel attack patterns
    
    ### Explainable AI with SHAP
    
    We use SHAP (SHapley Additive exPlanations) to interpret model predictions:
    
    - Identifies which words or phrases contribute to detection
    - Provides transparency for model decisions
    - Helps understand and improve model performance
    
    ### Technologies Used
    
    - TensorFlow/Keras for deep learning
    - Scikit-learn for traditional ML
    - SHAP for explainability
    - Streamlit for interactive interface
    
    ### References
    
    - [Understanding Prompt Injection Attacks](https://www.prompt-injection.com/)
    - [SHAP: Explainable AI](https://github.com/slundberg/shap)
    - [Detecting and Preventing LLM Prompt Injections](https://www.arxiv.org/abs/2302.12173)
    """)

# Note: This function is called from __init__.py, so no direct execution here
