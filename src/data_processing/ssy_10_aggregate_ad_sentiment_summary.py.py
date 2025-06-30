import os
import json
import re
import pandas as pd
from glob import glob
from collections import Counter

# === Path Settings ===
json_dir = "path/to/reddit_ad_data_json"
sentiment_csv = "path/to/reddit_comments_sentiment_analysis_3models.csv"

# === Load CSV and Clean Ad Name Keys ===
sentiment_df = pd.read_csv(sentiment_csv)

def clean_and_extract_key(text):
    """
    Clean ad name text into a normalized key by removing punctuation,
    lowering case, and joining the first 5 words with underscores.
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", "_", text.strip())  # Replace whitespace with underscore
    return "_".join(text.split("_")[:5])

sentiment_df['ad_name_key_cleaned'] = sentiment_df['Ad Name'].apply(clean_and_extract_key)

# === Extract Key from JSON Filename ===
def extract_json_key(file_path):
    """
    Extract a 5-word normalized key from the JSON filename.
    Skip leading digits (e.g., years like '2000', '2019').
    """
    filename = os.path.basename(file_path).replace(".json", "")
    parts = filename.lower().split("_")
    if parts[0].isdigit():
        parts = parts[1:]
    return "_".join(parts[:5])

# === Main Processing Logic ===
ad_sentiment_summary = []
json_files = glob(os.path.join(json_dir, "*.json"))

for json_file in json_files:
    try:
        json_key = extract_json_key(json_file)

        # Fuzzy match using partial key from filename
        matched = sentiment_df[
            sentiment_df['ad_name_key_cleaned'].str.contains(json_key, na=False)
        ]

        if matched.empty:
            continue

        # Aggregate sentiment counts
        sentiment_counts = matched['Final Sentiment'].value_counts().to_dict()
        total_positive = sentiment_counts.get('Positive', 0)
        total_negative = sentiment_counts.get('Negative', 0)
        total_neutral = sentiment_counts.get('Neutral', 0)
        total_comments = total_positive + total_negative + total_neutral
        final_sentiment = Counter(sentiment_counts).most_common(1)[0][0]

        ad_sentiment_summary.append({
            "ad_name": os.path.basename(json_file).replace(".json", ""),
            "Positive": total_positive,
            "Negative": total_negative,
            "Neutral": total_neutral,
            "Total Comments": total_comments,
            "Final Sentiment": final_sentiment
        })

    except Exception as e:
        print(f"Error processing file {json_file}: {e}")

# === Export Results ===
result_df = pd.DataFrame(ad_sentiment_summary)
result_path = os.path.join(os.getcwd(), "reddit_ad_sentiment_summary.csv")
result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f"✅ Sentiment aggregation complete. Results saved to: {result_path}")
