import os
import json
import random
import time
import signal
import sys
import socket
import requests
from yt_dlp import YoutubeDL
import logging
import urllib3
import fake_useragent  # Generate user agents

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AdvancedDownloader:
    def __init__(self):
        # User agent generator
        self.ua = fake_useragent.UserAgent()
        
        # Download log
        self.download_log = {
            "total_videos": 0,
            "success": [],
            "fail": []
        }
        self.output_dir = ""
        self.log_path = ""

    def check_internet_connection(self, host="8.8.8.8", port=53, timeout=5):
        """Enhanced network connection check"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except (socket.error, socket.timeout):
            return False

    def download_progress_hook(self, d):
        """Download progress callback"""
        if d['status'] == 'finished':
            print(f'\nDownload completed: {d["filename"]}')
        elif d['status'] == 'downloading':
            downloaded_bytes = d.get('downloaded_bytes', 0)
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            if total_bytes > 0:
                percent = downloaded_bytes * 100 / total_bytes
                print(f'\rDownload progress: {percent:.1f}%', end='', flush=True)

    def get_ydl_opts(self, video_id, output_dir):
        """Advanced download configuration"""
        return {
            'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            
            # Anti-crawling strategy
            'user_agent': self.ua.random,  # Random user agent
            'referer': 'https://www.youtube.com/',  # Add referrer
            
            # Download control
            'quiet': False,
            'no_warnings': True,
            'nooverwrites': True,
            'noplaylist': True,
            
            # Retry and interval
            'retries': 5,  # Increase retry count
            'fragment_retries': 3,
            'retry_sleep_functions': {
                'http': lambda n: min(n * 2, 10),  # Exponential backoff strategy
                'fragment': lambda n: min(n * 2, 10)
            },
            'sleep_interval_requests': random.uniform(1, 3),  # Random interval
            
            # Network and security
            'geo_bypass': True,
            'nocheckcertificate': True,
            'verify_ssl': False,
            
            # Progress callback
            'progress_hooks': [self.download_progress_hook],
        }

    def download_videos(self, video_data, json_filename):
        """Main download function"""
        # Create output directory
        base_output_dir = r"D:\youtube_vedio"
        output_dir = os.path.join(base_output_dir, json_filename, "videos")
        reports_dir = os.path.join(base_output_dir, json_filename, "reports")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.log_path = os.path.join(reports_dir, "download_report.json")
        self.download_log["total_videos"] = len(video_data)

        # Download loop
        for item in video_data:
            video_id = item['youtube_id']
            source_url = item['url']

            # Skip invalid YouTube IDs
            if not video_id or video_id == 'NaN':
                self.download_log["fail"].append({
                    "id": video_id, 
                    "url": source_url, 
                    "error": "Invalid YouTube ID"
                })
                continue
            
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                logging.info(f"▶ Downloading: {video_id}")

                with YoutubeDL(self.get_ydl_opts(video_id, output_dir)) as ydl:
                    ydl.download([url])

                logging.info(f"✅ Success: {video_id}")
                self.download_log["success"].append(video_id)

            except Exception as e:
                logging.error(f"❌ Failed: {video_id} - {str(e)}")
                self.download_log["fail"].append({
                    "id": video_id, 
                    "url": source_url, 
                    "error": str(e)
                })

            # Random delay
            time.sleep(random.uniform(3, 7))

        # Save download report
        self.save_download_report()

    def save_download_report(self):
        """Save download report"""
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.download_log, f, indent=2, ensure_ascii=False)

            logging.info(f"\n📄 Download report saved. Total videos: {self.download_log['total_videos']}")
            logging.info(f"Successfully downloaded: {len(self.download_log['success'])} videos")
            logging.info(f"Failed: {len(self.download_log['fail'])} videos")
            logging.info(f"Report path: {self.log_path}")
        except Exception as e:
            logging.error(f"Error saving report: {e}")

def main():
    import shutil

    def check_ffmpeg():
        if shutil.which("ffmpeg") is None:
            logging.error("⚠️ FFmpeg is not installed or not in PATH. Please install it to proceed.")
            sys.exit(1)

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load video data
    json_path = r"S:\Documents\2025 - Purdue\AAAAAPURDUE\5 Capstone\SuperBowlProject\data\raw\superbowl_ads\Youtube_ID_Yearly\2020.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load video data: {e}")
        sys.exit(1)

    # 从 JSON 文件名中提取文件名（不包含扩展名）
    json_filename = os.path.splitext(os.path.basename(json_path))[0]

    # 初始化下载器
    downloader = AdvancedDownloader()

    # 网络检查
    if not downloader.check_internet_connection():
        logging.error("Unable to connect to the internet")
        sys.exit(1)

    # 下载
    downloader.download_videos(video_data, json_filename)

if __name__ == "__main__":
    main()