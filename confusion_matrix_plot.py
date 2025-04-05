# confusion_matrix_plot.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from data_processing import load_and_preprocess_data


# Load saved model and label encoder
model = load_model("lstm_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

# Load and preprocess dataset
file_path = "PIA_Augmented_Dataset.csv"
X_padded, y = load_and_preprocess_data(file_path)

# Split test data
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Predict
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for LSTM Model")
plt.tight_layout()
plt.show()

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))
