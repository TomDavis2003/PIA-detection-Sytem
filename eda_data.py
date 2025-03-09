import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Perform Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("Dataset Head:\n", df.head())
    print("\nDataset Info:\n", df.info())
    print("\nClass Distribution:\n", df['Category'].value_counts())

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['Category'], palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Preprocess dataset
def preprocess_data(df):
    df['cleaned_text'] = df['Prompt'].apply(clean_text)
    return df

# Convert text to numerical vectors
def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['Category']  # Assuming the dataset has a 'label' column
    return X, y, vectorizer

# Train-test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    csv_path = "PIA_Augmented_Dataset.csv"  # Update with your actual dataset path
    df = load_data(csv_path)
    perform_eda(df)  # Perform EDA before preprocessing
    df = preprocess_data(df)
    X, y, vectorizer = vectorize_text(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data Preprocessing Complete.")
    print("X_train shape:", X_train.shape)
