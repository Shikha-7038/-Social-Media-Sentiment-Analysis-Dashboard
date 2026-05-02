"""
Text Cleaning and Preprocessing Module
Handles all text cleaning operations for sentiment analysis
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data (run once)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Additional custom stopwords (social media specific)
        self.custom_stopwords = {
            'rt', 'https', 'http', 'amp', 'user', 'im', 'ive',
            'u', 'ur', 'got', 'get', 'like', 'just', 'really'
        }
        self.stop_words.update(self.custom_stopwords)
    
    def clean_text(self, text):
        """
        Main cleaning function that applies all cleaning steps
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word, remove #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis (optional - keeps them as they can be useful)
        # text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def lemmatize_text(self, text):
        """Lemmatize words (convert to base form)"""
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Fully preprocessed text
        """
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Step 3: Lemmatization
        text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess entire dataframe column
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with cleaned column added
        """
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.preprocess)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def get_text_stats(self, df, text_column='cleaned_text'):
        """Get statistics about cleaned text"""
        df = df.copy()
        df['text_length'] = df[text_column].str.split().str.len()
        
        stats = {
            'total_samples': len(df),
            'avg_words_per_post': df['text_length'].mean(),
            'min_words': df['text_length'].min(),
            'max_words': df['text_length'].max(),
            'empty_texts': (df[text_column] == '').sum()
        }
        return stats

# Testing function
if __name__ == "__main__":
    # Test the cleaner
    cleaner = TextCleaner()
    
    test_texts = [
        "I LOVE this product! So amazing 😊 #blessed @user https://t.co/123",
        "This is terrible service, never buying again 😡",
        "Just an average experience, nothing special",
        "RT @someone: Best app ever!!! Highly recommended 👍👍👍"
    ]
    
    print("Testing Text Cleaner:")
    print("-" * 50)
    for text in test_texts:
        cleaned = cleaner.preprocess(text)
        print(f"Original: {text[:50]}...")
        print(f"Cleaned: {cleaned}")
        print("-" * 50)