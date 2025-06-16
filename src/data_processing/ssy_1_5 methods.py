import os
import json
import pandas as pd
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Disable TensorFlow backend support for transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Automatically select device: GPU (0) or CPU (-1)
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Sentiment analysis will run on: {'GPU' if device == 0 else 'CPU'}")

# Load RoBERTa sentiment model
def load_roberta_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

roberta_pipeline = load_roberta_pipeline()

def find_comments_directory():
    """
    Find the directory containing comment JSON files
    
    Returns:
        str: Path to comments directory
    """
    possible_paths = [
        'comments',
        'data/comments',
        'input/comments',
        os.path.join(os.path.dirname(__file__), 'comments')
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    raise FileNotFoundError("Could not find comments directory. Please create a 'comments' folder and add JSON files.")

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Method 1: TextBlob
def classify_textblob_sentiment(text):
    """
    Classify sentiment using TextBlob
    
    Args:
        text (str): Input text
    
    Returns:
        tuple: (sentiment label, polarity score)
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive", polarity
    elif polarity < -0.2:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Method 2: VADER
def classify_vader_sentiment(text):
    """
    Classify sentiment using VADER
    
    Args:
        text (str): Input text
    
    Returns:
        tuple: (sentiment label, compound score)
    """
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

# Method 3: RoBERTa
def classify_roberta_sentiment(text):
    """
    Classify sentiment using RoBERTa
    
    Args:
        text (str): Input text
    
    Returns:
        tuple: (sentiment label, sentiment value)
    """
    try:
        label = roberta_pipeline(text)[0]['label']
        if label == 'LABEL_2':
            return "Positive", 1
        elif label == 'LABEL_0':
            return "Negative", -1
        else:
            return "Neutral", 0
    except Exception:
        return "Neutral", 0

def process_sentiment_analysis():
    """
    Perform sentiment analysis on comments
    
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    # Store all results
    all_results = []

    # Find comments directory
    folder_path = find_comments_directory()
    output_path = "comments_sentiment_analysis.xlsx"

    # Iterate through all JSON files
    for filename in os.listdir(folder_path):
        if filename.endswith("_comments.json"):
            full_path = os.path.join(folder_path, filename)
            print(f"📄 Processing: {filename}")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry in data:
                        text = entry.get("text", "").strip()
                        if not text:
                            continue

                        tb_sentiment, tb_score = classify_textblob_sentiment(text)
                        vader_sentiment, vader_score = classify_vader_sentiment(text)
                        roberta_sentiment, _ = classify_roberta_sentiment(text)

                        # Majority voting
                        labels = [tb_sentiment, vader_sentiment, roberta_sentiment]
                        final_label = Counter(labels).most_common(1)[0][0]

                        all_results.append({
                            "Video ID": entry.get("video_id"),
                            "Comment": text,
                            "TextBlob Sentiment": tb_sentiment,
                            "TextBlob Score": tb_score,
                            "VADER Sentiment": vader_sentiment,
                            "VADER Score": vader_score,
                            "RoBERTa Sentiment": roberta_sentiment,
                            "Final Sentiment": final_label,
                            "Likes": entry.get("likes", 0),
                            "Author": entry.get("author"),
                            "Published At": entry.get("published_at")
                        })
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    # Export results to Excel
    df = pd.DataFrame(all_results)
    df.to_excel(output_path, index=False)
    print(f"\n✅ Analysis completed, results saved to: {output_path}")
    
    return df

def main():
    """
    Main function to run sentiment analysis
    """
    try:
        process_sentiment_analysis()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()