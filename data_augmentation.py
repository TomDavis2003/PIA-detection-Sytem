import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import os

nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("data/PIA Dataset.csv")

# Ensure dataset contains correct columns
if "Prompt" not in df.columns or "Category" not in df.columns:
    raise ValueError("Dataset must contain 'Prompt' and 'Category' columns!")

# Define augmentation functions
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word: 
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]  # Remove special characters
    if len(words) < 1:
        return text
    
    for _ in range(n):
        word_to_replace = random.choice(words)
        synonyms = get_synonyms(word_to_replace)
        if synonyms:
            synonym = random.choice(synonyms)
            words[words.index(word_to_replace)] = synonym
    return " ".join(words)

def random_deletion(text, p=0.2):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    if len(words) <= 1:
        return text  
    return " ".join([word for word in words if random.uniform(0, 1) > p])

def random_swap(text, n=1):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    if len(words) < 2:
        return text
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def random_insertion(text, n=1):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    if len(words) < 1:
        return text
    
    for _ in range(n):
        new_word = random.choice(words)
        synonyms = get_synonyms(new_word)
        if synonyms:
            synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, synonym)
    return " ".join(words)

# Apply augmentations
augmented_data = []
for index, row in df.iterrows():
    prompt, category = row["Prompt"], row["Category"]
    
    # Generate variations of the prompt
    augmented_data.append([prompt, category])  # Original
    augmented_data.append([synonym_replacement(prompt), category])
    augmented_data.append([random_deletion(prompt), category])
    augmented_data.append([random_swap(prompt), category])
    augmented_data.append([random_insertion(prompt), category])

# Create a new DataFrame and save
augmented_df = pd.DataFrame(augmented_data, columns=["Prompt", "Category"])
augmented_df.to_csv("PIA_Augmented_Dataset.csv", index=False)

print("✅ Data augmentation complete! Saved as 'PIA_Augmented_Dataset.csv'.")
