"""
Feature Extraction Module using TF-IDF
Converts text to numerical features for ML model
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os

class FeatureExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF Vectorizer
        
        Args:
            max_features (int): Maximum number of features to keep
            ngram_range (tuple): Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Use 1+log(tf)
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts (list or Series): List of text documents
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF features
        """
        # Convert to list if pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (list or Series): List of text documents
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names (top words)"""
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out()
        return []
    
    def save(self, filepath='models/tfidf_vectorizer.pkl'):
        """Save vectorizer to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath='models/tfidf_vectorizer.pkl'):
        """Load vectorizer from disk"""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")
    
    def get_feature_importance_preview(self, top_n=20):
        """
        Get preview of important features
        
        Args:
            top_n (int): Number of top features to show
            
        Returns:
            dict: Dictionary with top features by IDF score
        """
        if not self.is_fitted:
            return {}
        
        feature_names = self.get_feature_names()
        idf_values = self.vectorizer.idf_
        
        # Get top features by IDF (most rare but informative)
        top_indices = idf_values.argsort()[-top_n:][::-1]
        top_features = {feature_names[i]: idf_values[i] for i in top_indices}
        
        return top_features

# Testing
if __name__ == "__main__":
    # Test feature extractor
    extractor = FeatureExtractor(max_features=100)
    
    sample_texts = [
        "love amazing great fantastic good",
        "hate terrible bad awful worst",
        "average okay fine normal average"
    ]
    
    features = extractor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names: {extractor.get_feature_names()[:10]}")
    print(f"Top IDF features: {extractor.get_feature_importance_preview(5)}")