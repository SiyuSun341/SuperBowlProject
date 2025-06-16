import os
import json
import re
import pandas as pd
from collections import Counter

# === Custom meaningless words (can be expanded as needed) ===
custom_stopwords = set([
    'and', 'the', 'with', 'for', 'are', 'you', 'com', 'https', 'www', 'super',
    'bowl', 'watch', 'video', 'ads', 'from', 'that', 'have', 'has', 'more', 'this',
    'visit', 'like', 'just', 'click', 'link', 'get', 'out', 'about', 'see', 'all',
    'on', 'in', 'by', 'at', 'of', 'to', 'is', 'be', 'an', 'or', 'we', 'our', 'as',
    'it', 'not', 'if', 'can', 'make', 'up', 'who', 'a'
])

def find_input_directory():
    """
    Find input directory for details JSON files
    
    Returns:
        str: Path to details directory
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

def find_input_excel():
    """
    Find input Excel file
    
    Returns:
        str: Path to input Excel file
    """
    possible_paths = [
        'sentiment_positive_analysis.xlsx',
        'data/sentiment_positive_analysis.xlsx',
        'input/sentiment_positive_analysis.xlsx'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Could not find input Excel file")

def find_output_directory():
    """
    Find or create output directory
    
    Returns:
        str: Path to output directory
    """
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    try:
        # Find input directories and files
        json_folder = find_input_directory()
        video_ids_excel = find_input_excel()
        output_dir = find_output_directory()
        output_path = os.path.join(output_dir, 'title_description_top30_filtered_words.xlsx')

        # Step 1: Read Video ID column
        df_ids = pd.read_excel(video_ids_excel)
        if 'Video ID' not in df_ids.columns:
            raise ValueError("'Video ID' column not found in Excel, please check file format!")

        video_ids = df_ids['Video ID'].dropna().astype(str).tolist()

        # Step 2: Extract English words from title + description
        all_words = []

        for vid in video_ids:
            json_path = os.path.join(json_folder, f"{vid}_details.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        text = f"{data.get('title', '')} {data.get('description', '')}".lower()
                        words = re.findall(r'\b[a-z]{2,}\b', text)
                        filtered_words = [w for w in words if w not in custom_stopwords]
                        all_words.extend(filtered_words)
                except Exception as e:
                    print(f"⚠️ Error reading {json_path}: {e}")
            else:
                print(f"❌ Missing JSON: {json_path}")

        # Step 3: Count word frequency and output
        word_freq = Counter(all_words)
        top_30 = word_freq.most_common(30)

        df_top_words = pd.DataFrame(top_30, columns=["Word", "Frequency"])
        df_top_words.to_excel(output_path, index=False)

        print(f"\n✅ Top 30 high-frequency words saved to: {output_path}")
        print(df_top_words.head())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()