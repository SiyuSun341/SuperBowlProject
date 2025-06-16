import os
import json
import pandas as pd
from tqdm import tqdm

def find_input_file(filename):
    """
    Find input file dynamically
    
    Args:
        filename (str): Filename to search for
    
    Returns:
        str: Path to the input file
    """
    possible_paths = [
        filename,
        os.path.join('data', filename),
        os.path.join('input', filename)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find {filename}")

def find_json_directory():
    """
    Find JSON directory dynamically
    
    Returns:
        str: Path to the JSON directory
    """
    possible_paths = [
        'details',
        'data/details',
        'input/details'
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    raise FileNotFoundError("Could not find details directory")

def find_output_directory():
    """
    Find or create output directory
    
    Returns:
        str: Path to the output directory
    """
    output_dir = 'sentiment_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    try:
        # Find input files and directories
        csv_path = find_input_file('video_level_sentiment_summary.csv')
        json_folder = find_json_directory()
        output_dir = find_output_directory()

        # Paths for output files
        positive_output = os.path.join(output_dir, 'sentiment_positive_analysis.xlsx')
        negative_output = os.path.join(output_dir, 'sentiment_negative_analysis.xlsx')

        # Read CSV file
        df_sentiment = pd.read_csv(csv_path, encoding='utf-8-sig')
        df_sentiment.columns = df_sentiment.columns.str.strip()

        # Clean video name function
        def clean_video_name(x):
            if pd.isna(x):
                return None
            x = str(x)
            if x.startswith("="):
                return x.strip("=\"")  # Remove = and quotes
            return x.strip()

        df_sentiment['Video Name'] = df_sentiment['Video Name'].apply(clean_video_name)
        df_sentiment = df_sentiment[df_sentiment['Video Name'].notna()]

        # Select positive and negative video IDs
        positive_ids = set(df_sentiment[df_sentiment['Final Sentiment'].str.lower() == 'positive']['Video Name'])
        negative_ids = set(df_sentiment[df_sentiment['Final Sentiment'].str.lower() == 'negative']['Video Name'])

        # Load matching JSONs
        def load_matching_jsons(json_dir, ids_set):
            data = []
            for vid in tqdm(ids_set, desc="Loading JSONs"):
                file_path = os.path.join(json_dir, f"{vid}_details.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            data.append(content)
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
                else:
                    print(f"❌ Missing JSON file: {vid}_details.json")
            return data

        positive_data = load_matching_jsons(json_folder, positive_ids)
        negative_data = load_matching_jsons(json_folder, negative_ids)

        # Summarize video data
        def summarize_video(data_list, label):
            summary = []
            for d in data_list:
                summary.append({
                    "Video ID": d.get("video_id"),
                    "Title": d.get("title"),
                    "Channel": d.get("channel_name"),
                    "Views": d.get("view_count"),
                    "Likes": d.get("like_count"),
                    "Comments": d.get("comment_count"),
                    "Engagement Rate": d.get("engagement_rate"),
                    "Polarity": d.get("emotional_tone", {}).get("polarity_score"),
                    "Subjectivity": d.get("emotional_tone", {}).get("subjectivity_score"),
                    "Emotion Category": d.get("emotional_tone", {}).get("emotional_category"),
                    "Tags Count": len(d.get("tags", [])),
                    "Themes": ", ".join([k for k, v in d.get("potential_themes", {}).items() if v]),
                    "Sentiment Label": label
                })
            return pd.DataFrame(summary)

        df_pos = summarize_video(positive_data, "Positive")
        df_neg = summarize_video(negative_data, "Negative")

        # Save to Excel
        df_pos.to_excel(positive_output, index=False)
        df_neg.to_excel(negative_output, index=False)

        print(f"✅ Positive video analysis saved to: {positive_output}")
        print(f"✅ Negative video analysis saved to: {negative_output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()