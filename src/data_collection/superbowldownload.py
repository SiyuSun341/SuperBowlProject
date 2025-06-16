import yt_dlp
import os
import sys

def download_videos_from_superbowl_ads(url_file_path, output_dir='SuperBowlAds/'):
    """
    Batch download videos from a list of URLs from superbowl-ads.com.

    Args:
        url_file_path (str): Path to a text file containing superbowl-ads.com video URLs, one per line.
        output_dir (str): Directory to save videos. 
                          Will be created automatically if it doesn't exist.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created download directory: {output_dir}")

    # yt-dlp download options
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best', # Download best video and audio, then merge
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'), # Output file naming template
        'merge_output_format': 'mp4', # Merged format
        'noplaylist': True, # Ensure only single video is downloaded even if link is a playlist
        'retries': 5, # Number of retries on download failure
        'fragment_retries': 5, # Number of retries on fragment download failure
        # Optional download acceleration configuration (requires aria2c)
        # 'external_downloader': 'aria2c',
        # 'external_downloader_args': ['-x 16', '-k 1M'], # -x 16 means 16 connections, -k 1M means 1MB minimum chunk
        'ignoreerrors': True, # Skip errors and continue processing next URL
        'abort_on_error': False, # Do not terminate entire batch download due to single error
        'verbose': False, # Turn off verbose output for cleaner logs
        'progress': True, # Show download progress
    }

    video_urls = []
    try:
        with open(url_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'): # Ignore empty lines and comment lines
                    video_urls.append(url)
    except FileNotFoundError:
        print(f"Error: URL file '{url_file_path}' not found. Please ensure the file exists.")
        return

    if not video_urls:
        print(f"No valid URLs found in file '{url_file_path}'.")
        return

    print(f"Downloading {len(video_urls)} videos from '{url_file_path}' to '{output_dir}'")
    print("Note: Download may take some time, and some videos might not be downloadable (e.g., due to site changes or video removal).")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(video_urls)
        print("\n✅ All video download attempts completed!")
    except Exception as e:
        print(f"\n❌ Error during batch download: {e}")

def find_url_file():
    """
    Find the URL source file.

    Returns:
        str: Path to the URL source file
    """
    possible_paths = [
        'superbowl_download.txt',
        'superbowl_ads_urls.txt',
        os.path.join('data', 'superbowl_download.txt'),
        os.path.join('input', 'superbowl_download.txt')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Could not find URL source file. Please create a text file with video URLs.")

if __name__ == "__main__":
    # Default URL file path and download folder
    try:
        url_source_file = find_url_file()
        default_download_folder = 'SuperBowlAds'

        # Check command-line arguments
        if len(sys.argv) > 1:
            custom_download_folder = sys.argv[1]
            print(f"Detected custom output path: {custom_download_folder}")
        else:
            custom_download_folder = default_download_folder
            print(f"No output path specified, using default path: {default_download_folder}")

        download_videos_from_superbowl_ads(url_source_file, custom_download_folder)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a text file with video URLs and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")