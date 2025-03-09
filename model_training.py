from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

def train_random_forest(X_tfidf, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        print("Random Forest trained successfully.")
        return rf_model
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        return None

def train_lstm(X_padded, y, max_len, vocab_size=5000, num_classes=4):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
        lstm_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
            LSTM(64),
            Dense(num_classes, activation='softmax')
        ])
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        print("LSTM trained successfully.")
        return lstm_model
    except Exception as e:
        print(f"Error training LSTM: {e}")
        return None
