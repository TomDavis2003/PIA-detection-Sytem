import shap
import numpy as np

def shap_analysis_rf(rf_model, tfidf_vectorizer, prompt, X_tfidf):
    try:
        print("Starting Random Forest SHAP analysis...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_tfidf)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Debug: Check shapes
        print(f"X_tfidf shape: {X_tfidf.shape}, SHAP values shape: {np.array(shap_values).shape}")
        
        # Get SHAP values for the input prompt (single instance)
        shap_vals = shap_values[0]  # shap_values is [n_classes, n_samples, n_features], we want first sample
        feature_importance = sorted(zip(feature_names, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("RF SHAP analysis completed successfully.")
        return feature_importance
    except Exception as e:
        print(f"Error in RF SHAP analysis: {e}")
        return []

def shap_analysis_lstm(lstm_model, tokenizer, prompt, X_padded, max_len):
    try:
        print("Starting LSTM SHAP analysis...")

        # Check the model's expected input shape
        print(f"Model expected input shape: {lstm_model.input_shape}, Provided shape: {X_padded.shape}")

        # Ensure X_padded is shaped correctly (batch_size, max_len)
        if len(X_padded.shape) == 1:
            X_padded = np.expand_dims(X_padded, axis=0)  # Add batch dimension

        # Debug: Print shape after adjustment
        print(f"Adjusted X_padded shape: {X_padded.shape}")

        # Create a background dataset
        background = X_padded[:min(100, X_padded.shape[0])]

        # Convert to TensorFlow tensors if needed
        background = tf.convert_to_tensor(background, dtype=tf.float32)

        # Use DeepExplainer with the background
        explainer = shap.DeepExplainer(lstm_model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(tf.convert_to_tensor(X_padded, dtype=tf.float32))

        # Debug: Check output
        print(f"SHAP values computed, shape: {np.array(shap_values).shape}")

        # Extract SHAP values for the first sample
        shap_vals = shap_values[0][0]

        # Map SHAP values to words
        words = prompt.split()
        shap_vals = shap_vals[:len(words)] if len(shap_vals) > len(words) else np.pad(shap_vals, (0, len(words) - len(shap_vals)), 'constant')

        feature_importance = sorted(zip(words, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("LSTM SHAP analysis completed successfully.")
        return feature_importance
    except Exception as e:
        print(f"Error in LSTM SHAP analysis: {e}")
        return []
