"""
Sentiment Analysis Model Training Module
Trains and evaluates classification models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SentimentModel:
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize sentiment model
        
        Args:
            model_type (str): Type of model to use
                Options: 'logistic_regression', 'random_forest', 'naive_bayes'
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.is_trained = False
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"Model training completed: {self.model_type}")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report_dict['weighted avg']['precision'],
            'recall': report_dict['weighted avg']['recall'],
            'f1_score': report_dict['weighted avg']['f1-score'],
            'classification_report': report_dict,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save(self, filepath='models/sentiment_model.pkl'):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'classes': self.model.classes_ if self.is_trained else None
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/sentiment_model.pkl'):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def plot_confusion_matrix(cm, classes, save_path='outputs/figures/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Sentiment Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to avoid display issues
    print(f"Confusion matrix saved to {save_path}")

def plot_feature_importance(model, feature_names, top_n=20, save_path='outputs/figures/feature_importance.png'):
    """Plot feature importance for models that support it"""
    try:
        if hasattr(model.model, 'coef_'):
            # For logistic regression - get absolute importance across all classes
            importances = np.abs(model.model.coef_).mean(axis=0)
        elif hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        else:
            print("Model doesn't support feature importance")
            return
        
        # Ensure we don't request more features than available
        n_features = len(importances)
        actual_top_n = min(top_n, n_features)
        
        if actual_top_n == 0:
            print("No features available for importance plot")
            return
        
        # Get top features - ensure indices are within bounds
        indices = np.argsort(importances)[-actual_top_n:]
        
        # Make sure indices are valid and within feature_names range
        valid_indices = [i for i in indices if i < len(feature_names)]
        
        if len(valid_indices) == 0:
            print("No valid features found for importance plot")
            return
        
        top_features = [feature_names[i] for i in valid_indices]
        top_importances = importances[valid_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top {actual_top_n} Important Features - {model.model_type}')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to avoid display issues
        print(f"Feature importance plot saved to {save_path}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

# Main training pipeline
def run_training_pipeline():
    """Complete training pipeline"""
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load cleaned data
    print("\n1. Loading cleaned data...")
    
    # Check if cleaned data exists
    cleaned_data_path = 'data/processed/cleaned_sentiment_data.csv'
    if not os.path.exists(cleaned_data_path):
        print(f"   Error: {cleaned_data_path} not found!")
        print("   Please run data creation and cleaning first:")
        print("   python src/data_creation.py")
        print("   Then run the cleaning command")
        return None, None, None
    
    df = pd.read_csv(cleaned_data_path)
    print(f"   Loaded {len(df)} samples")
    print(f"   Class distribution:\n{df['sentiment'].value_counts()}")
    
    # 2. Prepare features and labels
    print("\n2. Preparing features and labels...")
    
    # Load vectorizer
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(vectorizer_path):
        print(f"   Error: {vectorizer_path} not found!")
        print("   Please run feature extraction first:")
        print("   python -c \"from src.feature_extractor import FeatureExtractor; import pandas as pd; df = pd.read_csv('data/processed/cleaned_sentiment_data.csv'); extractor = FeatureExtractor(); extractor.fit_transform(df['cleaned_text']); extractor.save()\"")
        return None, None, None
    
    extractor.load()
    
    # Transform texts
    X = extractor.transform(df['cleaned_text'])
    y = df['sentiment']
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Labels: {y.unique()}")
    
    # 3. Split data
    print("\n3. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # 4. Train multiple models and compare
    print("\n4. Training and comparing models...")
    models_to_test = ['logistic_regression', 'naive_bayes', 'random_forest']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n   Training {model_type}...")
        model = SentimentModel(model_type=model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    # 5. Select best model
    print("\n5. Selecting best model...")
    best_model_type = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_type]['accuracy']
    print(f"   Best model: {best_model_type} (Accuracy: {best_accuracy:.4f})")
    
    # 6. Train final model
    print(f"\n6. Training final {best_model_type} model...")
    final_model = SentimentModel(model_type=best_model_type)
    final_model.train(X_train, y_train)
    
    # 7. Evaluate final model
    print("\n7. Evaluating final model...")
    metrics = final_model.evaluate(X_test, y_test)
    print(f"\n   Final Model Performance:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, final_model.predict(X_test)))
    
    # 8. Plot confusion matrix
    print("\n8. Generating visualizations...")
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        classes=final_model.model.classes_
    )
    
    # 9. Plot feature importance (using all feature names, not sliced)
    feature_names = extractor.get_feature_names()
    print(f"   Total features: {len(feature_names)}")
    plot_feature_importance(final_model, feature_names, top_n=20)
    
    # 10. Save model
    print("\n9. Saving model...")
    final_model.save()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Return summary
    summary = {
        'best_model': best_model_type,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'test_size': X_test.shape[0],
        'train_size': X_train.shape[0]
    }
    
    return final_model, extractor, summary

if __name__ == "__main__":
    model, vectorizer, summary = run_training_pipeline()