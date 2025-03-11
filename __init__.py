from data_processing import load_and_preprocess_data
from model_training import train_random_forest, train_lstm
from frontend import run_frontend
import os

def main():
    # Load and preprocess data
    file_path = 'PIA_Augmented_Dataset.csv'
    data = load_and_preprocess_data(file_path)
    if data is None:
        return
    
    X_tfidf, X_padded, y, tfidf_vectorizer, tokenizer, label_encoder, max_len = data
    
    # Train models
    rf_model = train_random_forest(X_tfidf, y)
    lstm_model = train_lstm(X_padded, y, max_len, num_classes=len(label_encoder.classes_))
    
    if rf_model is None or lstm_model is None:
        return
    
    # Run Streamlit frontend
    print("\nStarting Prompt Injection Detection System with Streamlit...")
    run_frontend(rf_model, lstm_model, tfidf_vectorizer, tokenizer, label_encoder, max_len)

if __name__ == "__main__":
    main()
    # Run Streamlit app
    os.system("streamlit run __init__.py")
