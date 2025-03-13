import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from data_processing import load_and_preprocess_data

# Load and preprocess data
file_path = "PIA_Augmented_Dataset.csv"
X_padded, y = load_and_preprocess_data(file_path)
if X_padded is None:
    raise ValueError("Data preprocessing failed.")

# Train-test split
X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
# Boost 'Legitimate' weight (assuming class 0)
if np.unique(y_train)[0] == 0:
    class_weight_dict[0] *= 3.0

# --- LSTM ---
embedding_dim = 128
lstm_units = 64
num_classes = len(np.unique(y))

lstm_model = Sequential([
    Embedding(input_dim=10000 + 1, output_dim=embedding_dim, input_length=100),
    SpatialDropout1D(0.3),
    LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3),
    Dense(num_classes, activation='softmax')
])
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train, epochs=3, batch_size=32, validation_data=(X_test_lstm, y_test),
               callbacks=[early_stopping], verbose=1, class_weight=class_weight_dict)

# Evaluate LSTM on test set
y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_lstm_classes)
print(f"LSTM Test Accuracy: {accuracy:.4f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_lstm_classes)
print("\nLSTM Confusion Matrix:")
print(conf_matrix)

# Optionally, print classification report for more detailed metrics
print("\nLSTM Classification Report:")
label_encoder = joblib.load("label_encoder.pkl")  # Load label encoder to get class names
class_names = label_encoder.classes_
print(classification_report(y_test, y_pred_lstm_classes, target_names=class_names))

# Save model
lstm_model.save("lstm_model.keras")
