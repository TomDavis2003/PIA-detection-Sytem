# shap_analysis.py

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import tempfile
import os
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

def shap_analysis_rf(model, tfidf_vectorizer, prompt_text, tfidf_features):
    """
    Generate SHAP analysis for Random Forest model prediction
    
    Args:
        model: Trained Random Forest model
        tfidf_vectorizer: TF-IDF vectorizer used for feature extraction
        prompt_text: Original prompt text
        tfidf_features: TF-IDF features for the prompt
        
    Returns:
        List of tuples (feature, importance value) for top features
    """
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(tfidf_features)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get feature importance values
    if hasattr(shap_values, 'values'):
        importance_values = shap_values.values[0]
    else:
        importance_values = shap_values[0]
    
    # Get absolute importance for ranking
    abs_importance = np.abs(importance_values)
    
    # Get tokens in the prompt (for filtering)
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt_text.lower())
    prompt_tokens = set(cleaned_prompt.split())
    
    # Filter for features actually in the prompt
    feature_importance = []
    for idx, feature in enumerate(feature_names):
        if feature in prompt_tokens:
            feature_importance.append((feature, importance_values[idx]))
    
    # If not enough features from the prompt, add other top features
    if len(feature_importance) < 5:
        # Get indices of top features by absolute importance
        top_indices = np.argsort(-abs_importance)
        for idx in top_indices:
            feature = feature_names[idx]
            if (feature, importance_values[idx]) not in feature_importance:
                feature_importance.append((feature, importance_values[idx]))
                if len(feature_importance) >= 5:
                    break
    
    # Sort by absolute importance
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Limit to top 5
    return feature_importance[:5]
    
    # Generate and display SHAP plot in Streamlit
    if st.checkbox("Show RF SHAP Summary Plot"):
        # Create a temporary file for the plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Feature Contribution to Prediction")
            plt.tight_layout()
            plt.savefig(tmp_file.name)
            plt.close()
            
            # Display the plot
            st.image(tmp_file.name)
            # Clean up the temporary file
            os.unlink(tmp_file.name)
    
    return feature_importance[:5]

def shap_analysis_lstm(model, tokenizer, prompt_text, padded_sequence, max_len):
    """
    Generate SHAP analysis for LSTM model prediction
    
    Args:
        model: Trained LSTM model
        tokenizer: Keras tokenizer used for tokenization  
        prompt_text: Original prompt text
        padded_sequence: Padded sequence representation of the prompt
        max_len: Maximum sequence length
        
    Returns:
        List of tuples (token, importance value) for top tokens
    """
    # Create a background dataset (zeros)
    background = np.zeros((1, max_len))
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(padded_sequence)
    
    # For classification models, shap_values is a list of arrays, one per class
    # We'll focus on the positive class (assuming binary classification)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Get model prediction to determine which class's SHAP values to use
        pred_class = model.predict(padded_sequence)[0].argmax()
        shap_values_to_use = shap_values[pred_class][0]
    else:
        shap_values_to_use = shap_values[0]
    
    # Extract word mapping from tokenizer
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    
    # Get the actual tokens for this example
    tokens = []
    for idx in padded_sequence[0]:
        if idx > 0:  # Skip padding tokens
            tokens.append(index_word.get(idx, "UNK"))
    
    # Create list of (token, shap_value) pairs
    token_importance = []
    
    # Only consider non-padding tokens
    for i, token in enumerate(tokens):
        if i < len(shap_values_to_use):
            token_importance.append((token, shap_values_to_use[i]))
    
    # Sort by absolute importance
    token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Limit to top 5
    return token_importance[:5]
    
    # Generate and display SHAP plot in Streamlit
    if st.checkbox("Show LSTM SHAP Plot"):
        # Find non-zero tokens in the sequence
        non_zero_indices = [i for i, idx in enumerate(padded_sequence[0]) if idx > 0]
        if non_zero_indices:
            # Create a temporary file for the plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                # Create bar plot of SHAP values for each token
                plt.figure(figsize=(10, 5))
                token_display = [index_word.get(padded_sequence[0][i], "UNK") for i in non_zero_indices]
                shap_display = [shap_values_to_use[i] for i in non_zero_indices]
                
                # Sort by absolute SHAP value
                sorted_indices = np.argsort(np.abs(shap_display))[::-1][:10]  # Top 10 tokens
                token_display = [token_display[i] for i in sorted_indices]
                shap_display = [shap_display[i] for i in sorted_indices]
                
                plt.barh(range(len(token_display)), shap_display, color=['red' if x < 0 else 'blue' for x in shap_display])
                plt.yticks(range(len(token_display)), token_display)
                plt.title("Top Tokens Influencing LSTM Prediction")
                plt.xlabel("SHAP Value (Impact on Prediction)")
                plt.tight_layout()
                plt.savefig(tmp_file.name)
                plt.close()
                
                # Display the plot
                st.image(tmp_file.name)
                # Clean up the temporary file
                os.unlink(tmp_file.name)
    
    return token_importance[:5]

def generate_full_shap_analysis(rf_model, lstm_model, X_tfidf, X_padded, y, tfidf_vectorizer, tokenizer, label_encoder, max_len, sample_size=1000):
    """
    Generate comprehensive SHAP analysis for both models using training data
    
    Args:
        rf_model: Trained Random Forest model
        lstm_model: Trained LSTM model
        X_tfidf: TF-IDF features
        X_padded: Padded sequences for LSTM
        y: Target labels
        tfidf_vectorizer: TF-IDF vectorizer
        tokenizer: Keras tokenizer
        label_encoder: Label encoder
        max_len: Maximum sequence length
        sample_size: Number of samples to use for SHAP analysis
        
    Returns:
        Dictionary with paths to saved plots
    """
    # Create directories for plots
    os.makedirs("./shap_plots/rf", exist_ok=True)
    os.makedirs("./shap_plots/lstm", exist_ok=True)
    
    # Sample data for SHAP analysis (full dataset may be too large)
    indices = np.random.choice(len(y), min(sample_size, len(y)), replace=False)
    X_tfidf_sample = X_tfidf[indices]
    X_padded_sample = X_padded[indices]
    y_sample = y[indices]
    
    results = {
        'rf_plots': [],
        'lstm_plots': []
    }
    
    # Random Forest SHAP Analysis
    print("Generating Random Forest SHAP analysis...")
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer(X_tfidf_sample)
    
    # Global feature importance plot for RF
    feature_names = tfidf_vectorizer.get_feature_names_out()
    plt.figure(figsize=(12, 10))
    shap.summary_plot(rf_shap_values, X_tfidf_sample, feature_names=feature_names, show=False)
    plt.title("Global Feature Importance (Random Forest)")
    plt.tight_layout()
    rf_summary_plot = "./shap_plots/rf/global_importance.png"
    plt.savefig(rf_summary_plot)
    plt.close()
    results['rf_plots'].append(rf_summary_plot)
    
    # Get top 20 features by importance for RF
    mean_abs_shap = np.abs(rf_shap_values.values).mean(0)
    top_indices = np.argsort(-mean_abs_shap)[:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        [feature_names[i] for i in top_indices][::-1],
        [mean_abs_shap[i] for i in top_indices][::-1]
    )
    plt.title("Top 20 Features by Importance (Random Forest)")
    plt.xlabel("Mean |SHAP Value|")
    rf_top20_plot = "./shap_plots/rf/top20_features.png"
    plt.savefig(rf_top20_plot)
    plt.close()
    results['rf_plots'].append(rf_top20_plot)
    
    # LSTM SHAP Analysis (more complex due to sequential nature)
    print("Generating LSTM SHAP analysis...")
    # Use a smaller sample for LSTM due to computational intensity
    lstm_sample_size = min(200, len(indices))
    X_padded_lstm_sample = X_padded_sample[:lstm_sample_size]
    
    # Create a background dataset
    background = X_padded_sample[:100]  # Use 100 samples as background
    
    lstm_explainer = shap.DeepExplainer(lstm_model, background)
    lstm_shap_values = lstm_explainer.shap_values(X_padded_lstm_sample)
    
    # For classification, we may get a list of arrays (one per class)
    if isinstance(lstm_shap_values, list) and len(lstm_shap_values) > 1:
        # We'll focus on the positive class (assuming binary classification)
        lstm_shap_values_pos = lstm_shap_values[1]  # For the second class (e.g., "Malicious")
    else:
        lstm_shap_values_pos = lstm_shap_values
    
    # Calculate mean absolute SHAP value for each position in the sequence
    mean_abs_lstm_shap = np.abs(lstm_shap_values_pos).mean(axis=0)
    
    # Plot mean impact by sequence position
    plt.figure(figsize=(14, 6))
    plt.plot(range(max_len), mean_abs_lstm_shap)
    plt.title("Mean Token Impact by Position in Sequence (LSTM)")
    plt.xlabel("Token Position")
    plt.ylabel("Mean |SHAP Value|")
    lstm_position_plot = "./shap_plots/lstm/position_importance.png"
    plt.savefig(lstm_position_plot)
    plt.close()
    results['lstm_plots'].append(lstm_position_plot)
    
    # Create index to word mapping
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    
    # Identify most important tokens across all samples
    token_importance = {}
    for sample_idx in range(lstm_sample_size):
        for pos in range(max_len):
            token_id = X_padded_lstm_sample[sample_idx][pos]
            if token_id > 0:  # Not padding
                token = index_word.get(token_id, f"UNK_{token_id}")
                shap_val = lstm_shap_values_pos[sample_idx][pos]
                if token in token_importance:
                    token_importance[token].append(shap_val)
                else:
                    token_importance[token] = [shap_val]
    
    # Calculate mean absolute SHAP value for each token
    mean_token_importance = {}
    for token, values in token_importance.items():
        mean_token_importance[token] = np.mean(np.abs(values))
    
    # Get top 20 tokens by importance
    top_tokens = sorted(mean_token_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh([t[0] for t in top_tokens][::-1], [t[1] for t in top_tokens][::-1])
    plt.title("Top 20 Tokens by Importance (LSTM)")
    plt.xlabel("Mean |SHAP Value|")
    lstm_top20_plot = "./shap_plots/lstm/top20_tokens.png"
    plt.savefig(lstm_top20_plot)
    plt.close()
    results['lstm_plots'].append(lstm_top20_plot)
    
    print("SHAP analysis complete!")
    return results
