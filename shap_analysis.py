import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
import time
import io  # Added for in-memory file handling
from tensorflow.keras.preprocessing.sequence import pad_sequences

def shap_analysis_lstm(model, tokenizer, prompt_text, padded_sequence, max_len):
    """
    Generate SHAP analysis for LSTM model prediction
    """
    try:
        background = np.zeros((1, max_len))
        
        def model_predict(x):
            return model.predict(x, verbose=0)
        
        explainer = shap.KernelExplainer(model_predict, background, nsamples=50)
        shap_values = explainer.shap_values(padded_sequence)
        
        st.write(f"LSTM SHAP Values Type: {type(shap_values)}")
        if isinstance(shap_values, list):
            st.write(f"LSTM SHAP Values Length: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                st.write(f"Class {i} SHAP Shape: {sv.shape if sv is not None else 'None'}")
        else:
            st.write(f"LSTM SHAP Values Shape: {shap_values.shape if shap_values is not None else 'None'}")
        
        if isinstance(shap_values, list) and len(shap_values) == model.output_shape[-1]:
            shap_values_aggregated = np.mean(np.abs(shap_values), axis=0)[0]
        else:
            shap_values_aggregated = np.mean(np.abs(shap_values[0]), axis=0) if isinstance(shap_values, list) else np.mean(np.abs(shap_values), axis=2)[0]
        
        word_index = tokenizer.word_index
        index_word = {v: k for k, v in word_index.items()}
        
        tokens = [index_word.get(idx, "UNK") for idx in padded_sequence[0] if idx > 0]
        valid_positions = [i for i, idx in enumerate(padded_sequence[0]) if idx > 0]
        
        token_importance = []
        for i, pos in enumerate(valid_positions):
            if i < len(shap_values_aggregated):
                token_importance.append((tokens[i], float(shap_values_aggregated[pos])))
        
        token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        token_importance = token_importance[:5]
        
        if valid_positions:
            with io.BytesIO() as output:
                plt.figure(figsize=(10, 5))
                token_display = [index_word.get(padded_sequence[0][i], "UNK") for i in valid_positions]
                shap_display = [shap_values_aggregated[i] for i in valid_positions]
                sorted_indices = np.argsort(np.abs(shap_display))[::-1][:10]
                token_display = [token_display[i] for i in sorted_indices]
                shap_display = [shap_display[i] for i in sorted_indices]
                plt.barh(range(len(token_display)), shap_display, color=['red' if x < 0 else 'blue' for x in shap_display])
                plt.yticks(range(len(token_display)), token_display)
                plt.title("Top Tokens Influencing LSTM Prediction")
                plt.tight_layout()
                plt.savefig(output, format='png')
                output.seek(0)
                plt.close()
                st.image(output)
        
        return token_importance
    except Exception as e:
        st.error(f"Error in LSTM SHAP analysis: {str(e)}")
        preds = model.predict(padded_sequence, verbose=0)[0]
        st.write("Fallback: LSTM Prediction Contributions")
        st.bar_chart(dict(zip(["Legitimate", "Cmd Obfuscation", "Ctx Alteration", "Instr Hijacking"], preds)))
        return []

def generate_full_shap_analysis(lstm_model, X_padded, y, tokenizer, label_encoder, max_len, sample_size=1000):
    os.makedirs("./shap_plots/lstm", exist_ok=True)
    
    indices = np.random.choice(len(y), min(sample_size, len(y)), replace=False)
    X_padded_sample = X_padded[indices]
    y_sample = y[indices]
    
    results = {'lstm_plots': []}
    
    print("Generating LSTM SHAP analysis...")
    lstm_sample_size = min(200, len(indices))
    X_padded_lstm_sample = X_padded_sample[:lstm_sample_size]
    background = X_padded_sample[:100]
    
    def lstm_predict(x):
        return lstm_model.predict(x, verbose=0)
    
    lstm_explainer = shap.KernelExplainer(lstm_predict, background, nsamples=50)
    lstm_shap_values = lstm_explainer.shap_values(X_padded_lstm_sample[:1])
    
    if isinstance(lstm_shap_values, list) and len(lstm_shap_values) > 1:
        lstm_shap_values_pos = lstm_shap_values[0]
    else:
        lstm_shap_values_pos = lstm_shap_values
    
    mean_abs_lstm_shap = np.abs(lstm_shap_values_pos).mean(axis=0)
    plt.figure(figsize=(14, 6))
    plt.plot(range(max_len), mean_abs_lstm_shap)
    plt.title("Mean Token Impact by Position in Sequence (LSTM)")
    plt.xlabel("Token Position")
    plt.ylabel("Mean |SHAP Value|")
    lstm_position_plot = "./shap_plots/lstm/position_importance.png"
    plt.savefig(lstm_position_plot)
    plt.close()
    results['lstm_plots'].append(lstm_position_plot)
    
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    token_importance = {}
    for sample_idx in range(lstm_sample_size):
        for pos in range(max_len):
            token_id = X_padded_lstm_sample[sample_idx][pos]
            if token_id > 0:
                token = index_word.get(token_id, f"UNK_{token_id}")
                shap_val = lstm_shap_values_pos[sample_idx][pos]
                token_importance[token] = token_importance.get(token, []) + [shap_val]
    
    mean_token_importance = {k: np.mean(np.abs(v)) for k, v in token_importance.items()}
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
