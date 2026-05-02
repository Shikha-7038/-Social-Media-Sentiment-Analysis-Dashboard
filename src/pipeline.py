"""
Complete Pipeline Runner - Handles encoding issues on Windows
"""

import subprocess
import sys
import os
import io

# Fix console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def run_step(step_name, command):
    """Run a pipeline step with proper error handling"""
    print("\n" + "=" * 60)
    print(f"▶ {step_name}")
    print("=" * 60)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout[-500:])  # Print last 500 chars
        return True
    else:
        print("❌ ERROR")
        print(result.stderr)
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     SOCIAL MEDIA SENTIMENT ANALYSIS PIPELINE                ║
    ║                   (Windows Optimized)                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Create data
    if not run_step("Creating synthetic dataset", "python -X utf8 src/data_creation.py"):
        print("\nData creation failed. Check the error above.")
        return
    
    # Step 2: Clean data
    if not run_step("Cleaning and preprocessing data", 
                    'python -c "import pandas as pd; from src.text_cleaner import TextCleaner; df = pd.read_csv(\'data/raw/synthetic_sentiment_data.csv\', encoding=\'utf-8\'); cleaner = TextCleaner(); df_clean = cleaner.preprocess_dataframe(df, \'text\'); df_clean.to_csv(\'data/processed/cleaned_sentiment_data.csv\', index=False, encoding=\'utf-8\'); print(f\"Cleaned {len(df_clean)} samples\")"'):
        print("\nData cleaning failed.")
        return
    
    # Step 3: Feature extraction
    if not run_step("Extracting TF-IDF features",
                    'python -c "import pandas as pd; from src.feature_extractor import FeatureExtractor; df = pd.read_csv(\'data/processed/cleaned_sentiment_data.csv\', encoding=\'utf-8\'); extractor = FeatureExtractor(); extractor.fit_transform(df[\'cleaned_text\']); extractor.save(); print(\"Feature extraction complete\")"'):
        print("\nFeature extraction failed.")
        return
    
    # Step 4: Train model
    if not run_step("Training sentiment model", "python -X utf8 src/train_model.py"):
        print("\nModel training failed.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext step: Run the dashboard")
    print("streamlit run app/dashboard.py")
    print("OR")
    print("python run_dashboard.py")

if __name__ == "__main__":
    main()