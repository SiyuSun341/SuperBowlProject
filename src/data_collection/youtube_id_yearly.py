import pandas as pd
import os
import json

def split_youtube_data_by_year(input_csv_path, output_dir):
    """
    Split YouTube data by year, preserving URL and ID
    
    :param input_csv_path: Input CSV file path
    :param output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(input_csv_path)

    # Ensure DataFrame contains necessary columns
    required_columns = ['year', 'url', 'youtube_id']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the input CSV: {missing_columns}")

    # Group by year and create JSON files
    for year, year_df in df.groupby('year'):
        # Select URL and youtube_id columns
        year_data = year_df[['url', 'youtube_id']].to_dict('records')
        
        # Create output file path
        output_file_path = os.path.join(output_dir, f'{year}_youtube_data.json')
        
        # Write data to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(year_data, f, indent=2, ensure_ascii=False)

    print(f"JSON files have been created in {output_dir}")

def find_input_csv():
    """
    Find the input CSV file with YouTube data
    
    Returns:
        str: Path to the input CSV file
    """
    possible_paths = [
        'SuperBowl_Ads_Links_with_youtube_ids.csv',
        os.path.join('data', 'raw', 'superbowl_ads', 'SuperBowl_Ads_Links_with_youtube_ids.csv'),
        os.path.join('input', 'SuperBowl_Ads_Links_with_youtube_ids.csv')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Could not find input CSV file with YouTube data.")

def find_or_create_output_dir():
    """
    Find or create an output directory for yearly YouTube data
    
    Returns:
        str: Path to the output directory
    """
    possible_dirs = [
        'Youtube_ID_Yearly',
        os.path.join('data', 'raw', 'superbowl_ads', 'Youtube_ID_Yearly'),
        os.path.join('output', 'Youtube_ID_Yearly')
    ]

    for directory in possible_dirs:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        if os.path.isdir(directory):
            return directory
    
    # Fallback to current directory
    os.makedirs('Youtube_ID_Yearly', exist_ok=True)
    return 'Youtube_ID_Yearly'

def main():
    try:
        # Find input CSV file
        input_csv_path = find_input_csv()
        print(f"Using input CSV: {input_csv_path}")

        # Find or create output directory
        output_dir = find_or_create_output_dir()
        print(f"Output directory: {output_dir}")

        # Call function to split data
        split_youtube_data_by_year(input_csv_path, output_dir)
        
        print("Data splitting completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input CSV file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()