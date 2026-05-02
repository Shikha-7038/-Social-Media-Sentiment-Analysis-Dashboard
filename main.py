"""
Main orchestration script
Runs the complete pipeline from data creation to dashboard
"""

import subprocess
import sys
import os
import argparse

def run_command(command, description):
    """Run a shell command and print status"""
    print("\n" + "=" * 60)
    print(f"▶ {description}")
    print("=" * 60)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ SUCCESS")
        print(result.stdout)
    else:
        print("❌ ERROR")
        print(result.stderr)
        sys.exit(1)
    return result

def create_directories():
    """Create required directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src',
        'models',
        'app',
        'outputs/figures',
        'outputs/logs',
        'images',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created")

def main():
    parser = argparse.ArgumentParser(description='Social Media Sentiment Analysis Pipeline')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'data', 'clean', 'features', 'train', 'dashboard'],
                        help='Which pipeline step to run')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     SOCIAL MEDIA SENTIMENT ANALYSIS DASHBOARD               ║
    ║                    Complete Pipeline                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    create_directories()
    
    if args.step in ['all', 'data']:
        # Step 1: Create synthetic dataset
        run_command(
            "python src/data_creation.py",
            "Creating synthetic dataset"
        )
    
    if args.step in ['all', 'clean']:
        # Step 2: Clean and preprocess data
        run_command(
            "python -c \"import pandas as pd; from src.text_cleaner import TextCleaner; df = pd.read_csv('data/raw/synthetic_sentiment_data.csv'); cleaner = TextCleaner(); df_clean = cleaner.preprocess_dataframe(df, 'text'); df_clean.to_csv('data/processed/cleaned_sentiment_data.csv', index=False); print(f'Cleaned {len(df_clean)} samples')\"",
            "Cleaning and preprocessing data"
        )
    
    if args.step in ['all', 'features']:
        # Step 3: Extract features
        run_command(
            "python -c \"import pandas as pd; from src.feature_extractor import FeatureExtractor; df = pd.read_csv('data/processed/cleaned_sentiment_data.csv'); extractor = FeatureExtractor(); extractor.fit_transform(df['cleaned_text']); extractor.save()\"",
            "Extracting TF-IDF features"
        )
    
    if args.step in ['all', 'train']:
        # Step 4: Train model
        run_command(
            "python src/train_model.py",
            "Training sentiment model"
        )
    
    if args.step in ['all', 'dashboard']:
        # Step 5: Launch dashboard
        print("\n" + "=" * 60)
        print("▶ Launching Streamlit Dashboard")
        print("=" * 60)
        print("\n🚀 Starting dashboard...")
        print("📊 Open your browser and go to: http://localhost:8501")
        print("⏹ Press Ctrl+C to stop the dashboard\n")
        
        subprocess.run("streamlit run app/dashboard.py", shell=True)

if __name__ == "__main__":
    main()