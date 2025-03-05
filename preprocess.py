import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def load_and_preprocess_data():
    # Download required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

    # Load dataset
    try:
        df = pd.read_csv("data/prompt_data.csv")
        print("DataFrame columns:", df.columns.tolist())  # Debug print to see available columns
        
        # Assuming your text column might have a different name
        text_column = 'text'  # Change this to match your actual column name
        if text_column not in df.columns:
            # Try to find the text column by checking common names
            possible_names = ['text', 'prompt', 'content', 'message', 'input']
            for name in possible_names:
                if name in df.columns:
                    text_column = name
                    break
            else:
                raise KeyError(f"Could not find text column. Available columns: {df.columns.tolist()}")
        
        # Drop missing values and duplicates
        df.dropna(inplace=True)
        df.drop_duplicates(subset=[text_column], keep='first', inplace=True)
        
        # Clean text data
        print("Cleaning text data...")
        df['cleaned_prompt'] = df[text_column].apply(clean_text)
        
        # Encode labels
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])
        
        # Convert text into numerical features using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        X = vectorizer.fit_transform(df["cleaned_prompt"]).toarray()
        y = df["label"]
        
        # Split dataset with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        # Save the vectorizer and label encoder
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("\nPlease check your CSV file structure and column names.")
        raise

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data Preprocessing Complete!")
