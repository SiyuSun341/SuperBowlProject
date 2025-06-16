import os
import pandas as pd
from collections import Counter

def find_input_file():
    """
    Find the input CSV file for sentiment analysis
    
    Returns:
        str: Path to the input CSV file
    """
    possible_paths = [
        'comments_sentiment_analysis.csv',
        'data/comments_sentiment_analysis.csv',
        'input/comments_sentiment_analysis.csv',
        os.path.join(os.path.dirname(__file__), 'comments_sentiment_analysis.csv')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Could not find input CSV file for sentiment analysis.")

def create_output_directory():
    """
    Create a directory for output files
    
    Returns:
        str: Path to the output directory
    """
    output_dir = 'sentiment_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def analyze_video_sentiment(input_path):
    """
    Analyze sentiment at the video level
    
    Args:
        input_path (str): Path to the input CSV file
    
    Returns:
        pd.DataFrame: Video-level sentiment summary
    """
    # Read data
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    # Video-level sentiment distribution
    video_stats = []

    for video_id, group in df.groupby('Video Name'):
        sentiment_counts = group['Final Sentiment'].value_counts()
        total = sentiment_counts.sum()

        video_stats.append({
            "Video Name": video_id,
            "Positive": sentiment_counts.get('Positive', 0),
            "Negative": sentiment_counts.get('Negative', 0),
            "Neutral": sentiment_counts.get('Neutral', 0),
            "Total Comments": total,
            "Final Sentiment": sentiment_counts.idxmax()  # Most frequent sentiment
        })

    return pd.DataFrame(video_stats)

def compare_model_accuracies(input_path):
    """
    Compare accuracies of different sentiment analysis models
    
    Args:
        input_path (str): Path to the input CSV file
    
    Returns:
        pd.DataFrame: Model accuracy comparison
    """
    # Read data
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    # Model columns to compare
    model_columns = [
        'TextBlob Sentiment',
        'VADER Sentiment',
        'BART Sentiment',
        'RoBERTa Sentiment',
        'FinBERT Sentiment'
    ]

    accuracies = []

    for model in model_columns:
        match_count = (df[model] == df['Final Sentiment']).sum()
        total_count = df[model].notna().sum()
        accuracy = match_count / total_count if total_count > 0 else 0

        accuracies.append({
            'Model': model,
            'Matched Count': match_count,
            'Total Count': total_count,
            'Accuracy': round(accuracy, 4)
        })

    return pd.DataFrame(accuracies)

def main():
    """
    Main function to perform sentiment analysis and save results
    """
    try:
        # Find input file
        input_path = find_input_file()
        print(f"Using input file: {input_path}")

        # Create output directory
        output_dir = create_output_directory()
        print(f"Output directory: {output_dir}")

        # Video-level sentiment summary
        video_summary_df = analyze_video_sentiment(input_path)
        video_summary_output = os.path.join(output_dir, 'video_level_sentiment_summary.csv')
        video_summary_df.to_csv(video_summary_output, index=False, encoding='utf-8-sig')

        # Model accuracy comparison
        accuracy_df = compare_model_accuracies(input_path)
        model_accuracy_output = os.path.join(output_dir, 'model_accuracy_comparison.csv')
        accuracy_df.to_csv(model_accuracy_output, index=False, encoding='utf-8-sig')

        # Completion messages
        print(f"\n✅ Video sentiment statistics saved to: {video_summary_output}")
        print(f"✅ Model accuracy comparison saved to: {model_accuracy_output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input CSV file exists in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()