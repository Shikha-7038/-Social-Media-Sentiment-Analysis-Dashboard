# 📊 Social Media Sentiment Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements a **real-time sentiment analysis dashboard** that automatically classifies social media posts as **Positive**, **Negative**, or **Neutral** using Machine Learning and Natural Language Processing (NLP).

### ✨ Features

- 🔮 **Real-time Sentiment Prediction** - Instant analysis of any text input
- 📁 **Batch Processing** - Upload CSV files for bulk analysis
- 📊 **Interactive Visualizations** - Dynamic charts and graphs
- 📈 **Trend Analysis** - Track sentiment changes over time
- 🔍 **Word Frequency Analysis** - See most common words per sentiment
- 💡 **Business Insights** - Actionable recommendations based on data

## 🏗️ Architecture
Input Text → Cleaning → TF-IDF → Logistic Regression → Sentiment → Dashboard

text

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TextBlob |
| ML Model | Logistic Regression |
| Feature Extraction | TF-IDF |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |

## 📁 Project Structure

Social-Media-Sentiment-Analysis-Dashboard/
├── data/ # Datasets (raw and processed)
├── src/ # Source code modules
│ ├── data_creation.py
│ ├── text_cleaner.py
│ ├── feature_extractor.py
│ └── train_model.py
├── app/ # Streamlit dashboard
│ └── dashboard.py
├── models/ # Saved models
├── outputs/ # Results and figures
├── images/ # Screenshots for README
├── requirements.txt # Dependencies
├── main.py # Orchestration script
└── README.md # Documentation


## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
Running the Project
bash
# Run complete pipeline (data creation → cleaning → training)
python main.py --step all

# Launch dashboard
streamlit run app/dashboard.py
📊 Dashboard Preview
https://images/dashboard_preview.png

📈 Model Performance
Metric	Score
Accuracy	85.67%
Precision	85.42%
Recall	85.67%
F1-Score	85.48%
https://outputs/figures/confusion_matrix.png

💼 Business Applications
Brand Reputation Monitoring - Track public sentiment in real-time

Customer Feedback Analysis - Automatically categorize reviews

Campaign Performance - Measure marketing campaign impact

Competitor Analysis - Compare sentiment across competitors

Product Launch Tracking - Monitor reactions to new products

📝 Key Learnings
This project demonstrates:

End-to-end ML project implementation

Text preprocessing and NLP techniques

Production-ready code structure

Interactive dashboard development

Version control with Git/GitHub