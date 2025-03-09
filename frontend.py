import streamlit as st
from data_processing import preprocess_new_prompt
from shap_analysis import shap_analysis_rf, shap_analysis_lstm

def run_frontend(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len):
    # Streamlit app title and description
    st.title("Prompt Injection Attack Detection System")
    st.write("Enter a prompt below to classify it as Legitimate or Malicious using Random Forest and LSTM models. SHAP analysis will explain the predictions.")

    # Input field for the prompt
    prompt = st.text_area("Enter your prompt here:", height=100)

    # Button to trigger prediction
    if st.button("Analyze Prompt"):
        if not prompt:
            st.warning("Please enter a prompt to analyze.")
        else:
            # Preprocess new prompt
            tfidf_features, padded_sequence = preprocess_new_prompt(prompt, tfidf_vectorizer, tokenizer, max_len)

            # Random Forest Prediction
            rf_probs = rf_model.predict_proba(tfidf_features)[0]
            rf_pred = label_encoder.inverse_transform([rf_probs.argmax()])[0]
            rf_legit_prob = rf_probs[label_encoder.transform(['Legitimate'])[0]] * 100
            rf_mal_prob = 100 - rf_legit_prob

            # LSTM Prediction
            lstm_probs = lstm_model.predict(padded_sequence, verbose=0)[0]
            lstm_pred = label_encoder.inverse_transform([lstm_probs.argmax()])[0]
            lstm_legit_prob = lstm_probs[label_encoder.transform(['Legitimate'])[0]] * 100
            lstm_mal_prob = 100 - lstm_legit_prob

            # Display Predictions
            st.subheader("Random Forest Prediction")
            st.write(f"**Prediction:** {rf_pred}")
            st.write(f"**Legitimate:** {rf_legit_prob:.2f}% | **Malicious:** {rf_mal_prob:.2f}%")

            st.subheader("LSTM Prediction")
            st.write(f"**Prediction:** {lstm_pred}")
            st.write(f"**Legitimate:** {lstm_legit_prob:.2f}% | **Malicious:** {lstm_mal_prob:.2f}%")

            # SHAP Analysis
            rf_shap = shap_analysis_rf(rf_model, tfidf_vectorizer, prompt, tfidf_features)
            lstm_shap = shap_analysis_lstm(lstm_model, tokenizer, prompt, padded_sequence, max_len)

            st.subheader("Random Forest SHAP Analysis (Top 5 Features)")
            for feature, value in rf_shap:
                st.write(f"{feature}: {value:.4f}")

            st.subheader("LSTM SHAP Analysis (Top 5 Features)")
            for feature, value in lstm_shap:
                st.write(f"{feature}: {value:.4f}")

    # Footer
    st.write("---")
    st.write("Built with Streamlit by [Your Name] for Prompt Injection Detection Project")

# Note: This function is called from __init__.py, so no direct execution here
