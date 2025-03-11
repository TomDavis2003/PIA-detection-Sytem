import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def load_and_preprocess_data(file_path):
    try:
        # Read dataset
        df = pd.read_csv(file_path)
        
        # Clean prompts
        df['Prompt'] = df['Prompt'].apply(clean_text)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Category'])
        
        # For Random Forest: TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = tfidf_vectorizer.fit_transform(df['Prompt']).toarray()
        
        # For LSTM: Tokenization and Padding
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['Prompt'])
        X_sequences = tokenizer.texts_to_sequences(df['Prompt'])
        max_len = 100  # Fixed sequence length
        X_padded = pad_sequences(X_sequences, maxlen=max_len)
        
        return (X_tfidf, X_padded, y, tfidf_vectorizer, tokenizer, label_encoder, max_len)
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None

def preprocess_new_prompt(prompt, tfidf_vectorizer, tokenizer, max_len):
    cleaned_prompt = clean_text(prompt)
    tfidf_features = tfidf_vectorizer.transform([cleaned_prompt]).toarray()
    sequence = tokenizer.texts_to_sequences([cleaned_prompt])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return tfidf_features, padded_sequence
