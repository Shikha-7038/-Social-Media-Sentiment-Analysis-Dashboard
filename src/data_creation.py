"""
Synthetic Social Media Dataset Creator
Creates realistic social media posts with sentiment labels
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import io

# Fix for Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Define sentiment word banks
positive_words = [
    # Product related
    'amazing', 'fantastic', 'excellent', 'perfect', 'wonderful', 'brilliant',
    'love', 'awesome', 'great', 'good', 'happy', 'satisfied', 'impressed',
    'best', 'incredible', 'outstanding', 'superb', 'fantastic', 'glad',
    # Service related
    'quick', 'responsive', 'helpful', 'friendly', 'professional', 'smooth',
    'easy', 'convenient', 'reliable', 'efficient', 'seamless',
    # Emotional
    'delighted', 'pleased', 'thrilled', 'grateful', 'thankful'
]

negative_words = [
    # Product related
    'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointed',
    'hate', 'useless', 'broken', 'defective', 'waste', 'pathetic',
    # Service related
    'slow', 'rude', 'unresponsive', 'frustrating', 'difficult', 'complicated',
    'expensive', 'overpriced', 'late', 'cancelled', 'ignored',
    # Emotional
    'angry', 'annoyed', 'frustrated', 'upset', 'regret'
]

neutral_words = [
    'okay', 'fine', 'average', 'decent', 'mediocre', 'alright',
    'regular', 'standard', 'typical', 'normal', 'usual', 'fair',
    'acceptable', 'moderate', 'so-so', 'neither', 'whatever'
]

# Social media post templates
templates = [
    "Just tried {product} and it was {adj}! {extra}",
    "{adj} experience with {product} today. {extra}",
    "{product} is {adj}! {extra}",
    "My review of {product}: {adj}. {extra}",
    "I {adj} {product}! {extra}",
    "{product} service was {adj}. {extra}",
    "Had a {adj} time with {product}. {extra}",
    "{product} - {adj}. {extra}",
    "Customer service at {product} is {adj}. {extra}",
    "The {product} app is {adj}. {extra}"
]

products = [
    'Netflix', 'Amazon', 'Spotify', 'Uber', 'Airbnb', 
    'GitHub', 'ChatGPT', 'Instagram', 'Twitter', 'YouTube',
    'Zomato', 'Swiggy', 'Flipkart', 'Myntra', 'Paytm'
]

# Emoji replacements (text-based for Windows compatibility)
emoji_map = {
    'positive': [' :) ', ' <3 ', ' :D ', ' ^_^ ', ' (y) ', ' great! ', ' awesome! '],
    'negative': [' :( ', ' :/ ', ' :| ', ' >:( ', ' ugh ', ' oh no ', ' yikes '],
    'neutral': [' :| ', ' meh ', ' okay ', ' i see ', ' hmm ']
}

hashtag_map = {
    'positive': ['#loveit', '#recommended', '#happy', '#best', '#amazing'],
    'negative': ['#badservice', '#disappointed', '#fail', '#worst', '#terrible'],
    'neutral': ['#meh', '#justokay', '#average', '#sososo', '#nodrama']
}

def add_emoji_text(sentiment):
    """Add text-based emoji replacements (no Unicode emojis)"""
    return random.choice(emoji_map[sentiment])

def add_hashtag(sentiment):
    """Add relevant hashtag"""
    return random.choice(hashtag_map[sentiment])

def generate_synthetic_post(sentiment):
    """Generate a single synthetic social media post"""
    template = random.choice(templates)
    product = random.choice(products)
    
    if sentiment == 'positive':
        adj = random.choice(positive_words)
        extra = random.choice([
            "Will definitely buy again!", "Highly recommended!",
            "Best decision ever!", "5 stars!",
            "Keep up the good work!", "Made my day better!",
            "This is what I call quality service!", "Thumbs up!"
        ])
    elif sentiment == 'negative':
        adj = random.choice(negative_words)
        extra = random.choice([
            "Never buying again.", "Waste of money!",
            "Disappointed.", "1 star!",
            "Fix your service!", "Complete failure.",
            "Don't waste your time here!", "Stay away!"
        ])
    else:  # neutral
        adj = random.choice(neutral_words)
        extra = random.choice([
            "It works.", "Not bad, not great.", "Average experience.",
            "Could be better.", "It's okay I guess.", "3 stars.",
            "Nothing special.", "As expected."
        ])
    
    post = template.format(product=product, adj=adj, extra=extra)
    
    # Add text-based emoji and hashtag with 40% probability
    if random.random() < 0.4:
        post += add_emoji_text(sentiment)
    
    if random.random() < 0.3:
        post += " " + add_hashtag(sentiment)
    
    return post

def create_synthetic_dataset(num_samples=3000):
    """Create complete synthetic dataset"""
    print("Creating synthetic dataset...")
    print(f"Target: {num_samples} samples")
    
    # Create balanced dataset
    samples_per_class = num_samples // 3
    remainder = num_samples % 3
    
    sentiments = ['positive'] * samples_per_class + \
                 ['negative'] * samples_per_class + \
                 ['neutral'] * samples_per_class
    
    # Add remainder samples to positive class
    sentiments.extend(['positive'] * remainder)
    random.shuffle(sentiments)
    
    dataset = []
    platforms = ['Twitter', 'YouTube', 'Instagram', 'Reddit']
    
    for i, sentiment in enumerate(sentiments):
        post = generate_synthetic_post(sentiment)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
        dataset.append({
            'text': post,
            'sentiment': sentiment,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'platform': random.choice(platforms),
            'likes': random.randint(0, 1000),
            'retweets': random.randint(0, 500)
        })
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"   Generated {i + 1}/{num_samples} samples...")
    
    df = pd.DataFrame(dataset)
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC SOCIAL MEDIA DATASET CREATOR")
    print("=" * 60)
    
    # Create dataset
    df = create_synthetic_dataset(3000)
    
    # Save to CSV
    import os
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/synthetic_sentiment_data.csv', index=False, encoding='utf-8')
    
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETED!")
    print("=" * 60)
    print(f"\nTotal samples created: {len(df)}")
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    print("\nPlatform distribution:")
    print(df['platform'].value_counts())
    
    # Sample preview (without emoji issues)
    print("\nSample data (first 5 rows):")
    print("-" * 60)
    for idx, row in df.head(5).iterrows():
        print(f"Text: {row['text'][:80]}...")
        print(f"Sentiment: {row['sentiment']}")
        print(f"Platform: {row['platform']}")
        print("-" * 40)