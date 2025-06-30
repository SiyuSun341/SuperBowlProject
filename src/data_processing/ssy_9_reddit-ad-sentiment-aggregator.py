import os
import json
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ========== Environment Configuration ==========
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Sentiment analysis will run on: {'GPU' if device == 0 else 'CPU'}")

# ========== Load RoBERTa Sentiment Model ==========
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if device == 0 else "cpu")
roberta_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

# ========== Initialize Traditional Sentiment Tools ==========
vader_analyzer = SentimentIntensityAnalyzer()

# ========== Define Sentiment Classification Functions ==========
def classify_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def classify_vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def classify_roberta_sentiment(text):
    try:
        label = roberta_pipeline(text)[0]['label']
        if label == 'LABEL_2':
            return "Positive"
        elif label == 'LABEL_0':
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

# ========== Path Settings ==========
folder_path = "path/to/reddit_ad_data_json"
output_file = os.path.join(folder_path, "reddit_ad_sentiment_summary.xlsx")

# ========== Main Aggregation Logic ==========
summary_data = []

for filename in os.listdir(folder_path):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(folder_path, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        print(f"❌ Failed to read {filename}")
        continue

    video_id = filename.replace(".json", "")
    reddit_discussions = data.get("RedditDiscussions", [])

    if not reddit_discussions:
        summary_data.append({
            "Video ID": video_id,
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0,
            "Total Comments": 0,
            "Final Sentiment": "No Comments"
        })
        continue

    pos, neg, neu = 0, 0, 0

    for discussion in reddit_discussions:
        texts = []

        if discussion.get("SubmissionTitle"):
            texts.append(discussion["SubmissionTitle"])
        if discussion.get("SubmissionText") and discussion["SubmissionText"] != "[No selftext]":
            texts.append(discussion["SubmissionText"])
        for comment in discussion.get("Comments", []):
            if comment.get("CommentText"):
                texts.append(comment["CommentText"])

        for text in texts:
            votes = [
                classify_textblob_sentiment(text),
                classify_vader_sentiment(text),
                classify_roberta_sentiment(text)
            ]
            majority_vote = Counter(votes).most_common(1)[0][0]
            if majority_vote == "Positive":
                pos += 1
            elif majority_vote == "Negative":
                neg += 1
            else:
                neu += 1

    total = pos + neg + neu
    final_sentiment = (
        max([("Positive", pos), ("Negative", neg), ("Neutral", neu)], key=lambda x: x[1])[0]
        if total > 0 else "No Comments"
    )

    summary_data.append({
        "Video ID": video_id,
        "Positive": pos,
        "Negative": neg,
        "Neutral": neu,
        "Total Comments": total,
        "Final Sentiment": final_sentiment
    })

# ========== Export Results ==========
df = pd.DataFrame(summary_data)
df.to_excel(output_file, index=False)
print(f"\n✅ Sentiment analysis completed. Results saved to: {output_file}")
