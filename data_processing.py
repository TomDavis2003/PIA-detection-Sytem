import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Prompt'] = df['Prompt'].apply(clean_text)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Category'])

        # Tokenization for LSTM
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(df['Prompt'])
        X_sequences = tokenizer.texts_to_sequences(df['Prompt'])
        max_len = 100
        X_padded = pad_sequences(X_sequences, maxlen=max_len)

        # Save preprocessors
        joblib.dump(tokenizer, "tokenizer.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        joblib.dump(max_len, "max_len.pkl")

        return X_padded, y  # Return only X_padded and y
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None

def preprocess_new_prompt(prompt, tokenizer, max_len):
    cleaned_prompt = clean_text(prompt)
    sequence = tokenizer.texts_to_sequences([cleaned_prompt])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence  # Return only padded_sequence
