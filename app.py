import streamlit as st
import pickle
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data

# Set page configuration
st.set_page_config(
    page_title="Prompt Injection Detector",
    page_icon="🚀",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_models():
    """Load all required models and return them"""
    try:
        # Load Random Forest model
        with open("models/random_forest.pkl", "rb") as f:
            rf_model = pickle.load(f)
            st.success("✅ Random Forest model loaded successfully")

        # Load LSTM model
        lstm_model = None
        lstm_model_path = "models/lstm_model.h5"
        
        try:
            lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
            st.success("✅ LSTM model loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ LSTM model loading failed: {str(e)}")
            
        # Load vectorizer
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            st.success("✅ Vectorizer loaded successfully")
            
        # Load label encoder
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
            st.success("✅ Label encoder loaded successfully")
            
        return rf_model, lstm_model, vectorizer, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error in model loading: {str(e)}")
        return None, None, None, None

def make_predictions(user_input, rf_model, lstm_model, vectorizer, label_encoder):
    """Make predictions using both models"""
    try:
        # Transform input
        input_vectorized = vectorizer.transform([user_input]).toarray()
        
        # Random Forest prediction
        rf_pred = rf_model.predict(input_vectorized)
        rf_prob = rf_model.predict_proba(input_vectorized)
        rf_label = label_encoder.inverse_transform(rf_pred)[0]
        rf_confidence = np.max(rf_prob) * 100
        
        # LSTM prediction
        lstm_label = None
        lstm_confidence = None
        if lstm_model is not None:
            lstm_pred = lstm_model.predict(input_vectorized)
            lstm_label = label_encoder.inverse_transform([np.argmax(lstm_pred)])[0]
            lstm_confidence = np.max(lstm_pred) * 100
            
        return {
            'rf_label': rf_label,
            'rf_confidence': rf_confidence,
            'lstm_label': lstm_label,
            'lstm_confidence': lstm_confidence,
            'input_vectorized': input_vectorized
        }
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        return None

def generate_shap_explanation(rf_model, input_vectorized, vectorizer):
    """Generate SHAP values explanation with proper visualization"""
    try:
        # Use TreeExplainer for Random Forest models
        explainer = shap.TreeExplainer(rf_model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(input_vectorized)
        
        # Handle multi-output models (e.g., binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use SHAP values for the positive class
        
        # Check if SHAP values exist
        if shap_values is None or len(shap_values) == 0:
            st.error("❌ SHAP values could not be computed. Check model compatibility.")
            return
        
        # Generate SHAP summary plot
        st.write("### SHAP Summary Plot - Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, input_vectorized, feature_names=vectorizer.get_feature_names_out(), show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error generating SHAP explanation: {str(e)}")

def main():
    st.title("🚀 Prompt Injection Attack Detection")
    st.write("This application detects potential prompt injection attacks in user inputs.")
    
    # Load all models
    rf_model, lstm_model, vectorizer, label_encoder = load_models()
    
    if not all([rf_model, vectorizer, label_encoder]):
        st.error("❌ Critical components failed to load. Please check the logs and try again.")
        return
    
    # User input section
    st.write("### Enter Your Prompt")
    user_input = st.text_area(
        "Type or paste your prompt here:",
        height=150,
        key="prompt_input"
    )
    
    # Analysis button
    if st.button("Analyze Prompt", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a prompt before analyzing.")
            return
            
        with st.spinner("Analyzing prompt..."):
            # Make predictions
            results = make_predictions(user_input, rf_model, lstm_model, vectorizer, label_encoder)
            
            if results:
                # Create two columns for the results
                col1, col2 = st.columns(2)
                
                # Random Forest results
                with col1:
                    st.write("### Random Forest Model")
                    st.write(f"**Prediction:** {results['rf_label']}")
                    st.write(f"**Confidence:** {results['rf_confidence']:.2f}%")
                
                # LSTM results
                with col2:
                    st.write("### LSTM Model")
                    if results['lstm_label']:
                        st.write(f"**Prediction:** {results['lstm_label']}")
                        st.write(f"**Confidence:** {results['lstm_confidence']:.2f}%")
                    else:
                        st.info("ℹ️ LSTM model not available")
                
                # Generate SHAP explanation
                generate_shap_explanation(
                    rf_model, 
                    results['input_vectorized'],
                    vectorizer
                )
                
    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit and ❤️")

if __name__ == "__main__":
    main()