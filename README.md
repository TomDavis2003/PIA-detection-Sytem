# Prompt Injection Attack Detection System using Explainable AI  

This project is a Prompt Injection Attack (PIA) Detection System using an LSTM model with SHAP-based explainability, built with Streamlit as the frontend. 

## Core Features

✅ Detects Prompt Injection Attacks using LSTM

✅ Provides explainability using SHAP

✅ Interactive Streamlit UI for real-time predictions

✅ Dataset visualization

## How to Run  

1. Clone the repository 
```sh
git clone https://github.com/Akshara-Balan/PIA-Detection-System-using-Explainable-AI.git
cd PIA-Detection-System-using-Explainable-AI
```

2. Install dependencies
```sh
pip install -r requirements.txt
```

3. Train and save the model
```sh
python model_training.py
```

4. Run the streamlit UI
   
   Windows:
   
   ```sh
   python -m streamlit run frontend.py
   ```

   Linux:

   ```sh
   streamlit run frontend.py
   ```
   
## How we did it

1. Created a dataset manually (PIA Dataset.csv) with 500 datas for each of the 4 categories.

2. Performed Data Augmentation using the snippet: data_augmentation.py , which gave PIA_Augmented_Dataset.csv with 2500 datas for each of the category.

3. Performed Exploratory Data Anlysis on this new dataset: eda_data.py

4. Trained the LSTM model with augmented data: model_training.py and saved the model.

5. Added SHAP analysis module for the saved model: shap_analysis.py

6. Created frontend module with streamlit framework: frontend.py

## How It All Works Together

### Training Phase (Offline)

* Train LSTM model on labeled PIA dataset.
* Save model (lstm_model.keras) and preprocessing artifacts.

### Inference & Explainability (Online - Streamlit)
* User inputs a prompt in the Streamlit UI.
* The prompt is preprocessed and fed to the trained LSTM model.
* The model predicts whether the prompt is legitimate or a PIA.
* SHAP analysis explains the prediction by highlighting influential words.


## Key Technologies
🔸 Deep Learning: TensorFlow (LSTM)

🔸 Explainability: SHAP

🔸 Frontend: Streamlit

🔸 Data Processing: Pandas, NumPy

🔸 Evaluation: Scikit-Learn
