import shap
import pickle
import numpy as np
from preprocess import load_and_preprocess_data

# Load models and data
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Apply SHAP to Random Forest
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
