import os
import json
import csv
import re
import time
import random
import requests
import logging
import urllib3
import pandas as pd
from typing import Dict, Any, List
from textblob import TextBlob
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# YouTube Transcript extraction
from youtube_transcript_api import YouTubeTranscriptApi

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('youtube_video_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeVideoAnalyzer:
    def __init__(self, 
                 api_key: str, 
                 input_file: str, 
                 output_dir: str,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        Initialize YouTube Video Analyzer
        
        :param api_key: YouTube Data API key
        :param input_file: JSON file with video information
        :param output_dir: Output directory for saving files
        :param max_retries: Maximum number of retries for requests
        :param retry_delay: Delay between retries
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'details'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comments'), exist_ok=True)
        
        # Initialize tools
        self.ua = UserAgent()
        
        # Configuration
        self.api_key = api_key
        self.input_file = input_file
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load video data
        self.video_data = self._load_video_data()
        
        # Error tracking
        self.failed_videos = []

    def _load_video_data(self) -> List[Dict[str, str]]:
        """
        Load video data from JSON file
        
        :return: List of video information
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            
            # Handle different input formats
            if isinstance(video_data[0], dict):
                # Format with URL and YouTube ID
                valid_videos = [
                    item for item in video_data 
                    if isinstance(item.get('youtube_id'), str)
                ]
            else:
                # Simple list of YouTube IDs
                valid_videos = [
                    {'youtube_id': vid, 'url': f'https://youtu.be/{vid}'} 
                    for vid in video_data 
                    if isinstance(vid, str)
                ]
            
            logger.info(f"Loaded {len(valid_videos)} valid videos")
            return valid_videos
        
        except Exception as e:
            logger.error(f"Error loading video data: {e}")
            return []

    def _create_robust_session(self) -> requests.Session:
        """
        Create a session with retry and timeout mechanisms
        
        :return: Requests session
        """
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        # Add adapters
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session

    def _get_headers(self) -> Dict[str, str]:
        """
        Generate realistic request headers
        
        :return: Headers dictionary
        """
        return {
            'User-Agent': self.ua.random,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }

    def get_video_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve video transcript
        
        :param video_id: YouTube video ID
        :return: List of transcript entries
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as e:
            logger.warning(f"Transcript retrieval failed for {video_id}: {e}")
            return []

    def get_youtube_video_details(self, video_item: Dict[str, str]) -> Dict[str, Any]:
        """
        Retrieve detailed information for a single YouTube video
        
        :param video_item: Video information dictionary
        :return: Dictionary of video details
        """
        video_id = video_item.get('youtube_id')
        source_url = video_item.get('url', 'N/A')
        
        if not video_id:
            logger.error(f"Invalid video ID for URL: {source_url}")
            self.failed_videos.append({
                'youtube_id': video_id,
                'source_url': source_url,
                'reason': 'Invalid or missing YouTube ID'
            })
            return {}
        
        # Create robust session
        session = self._create_robust_session()
        
        try:
            # Video details endpoint
            video_endpoint = "https://www.googleapis.com/youtube/v3/videos"
            channel_endpoint = "https://www.googleapis.com/youtube/v3/channels"
            
            # Retrieve video details
            video_params = {
                'part': 'snippet,contentDetails,statistics,topicDetails,localizations',
                'id': video_id,
                'key': self.api_key
            }
            video_response = session.get(
                video_endpoint, 
                params=video_params, 
                headers=self._get_headers(),
                timeout=10
            )
            video_data = video_response.json()
            
            if not video_data.get('items'):
                logger.warning(f"No details found for video {video_id}")
                self.failed_videos.append({
                    'youtube_id': video_id,
                    'source_url': source_url,
                    'reason': 'No video details found'
                })
                return {}
            
            video_item = video_data['items'][0]
            snippet = video_item.get('snippet', {})
            stats = video_item.get('statistics', {})
            
            # Retrieve channel details
            channel_params = {
                'part': 'snippet,statistics',
                'id': snippet.get('channelId', ''),
                'key': self.api_key
            }
            channel_response = session.get(
                channel_endpoint, 
                params=channel_params, 
                headers=self._get_headers(),
                timeout=10
            )
            channel_data = channel_response.json()
            
            # Extract channel information
            channel_info = channel_data['items'][0] if channel_data.get('items') else {}
            
            # Engagement rate calculation
            def calculate_engagement_rate(views, likes, comments):
                if views == 0:
                    return 0
                return (likes + comments) / views * 100
            
            # Duration parsing
            def parse_duration(duration: str) -> Dict[str, int]:
                match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
                if not match:
                    return {'hours': 0, 'minutes': 0, 'seconds': 0}
                
                hours = int(match.group(1)[:-1]) if match.group(1) else 0
                minutes = int(match.group(2)[:-1]) if match.group(2) else 0
                seconds = int(match.group(3)[:-1]) if match.group(3) else 0
                
                return {'hours': hours, 'minutes': minutes, 'seconds': seconds}
            
            # Theme extraction
            def extract_potential_themes(description: str) -> Dict[str, bool]:
                themes_keywords = {
                    'Humor': ['funny', 'comedy', 'joke', 'hilarious'],
                    'Emotional': ['touching', 'inspiring', 'heartwarming', 'motivational'],
                    'Technology': ['tech', 'innovation', 'future', 'advanced'],
                    'Social Issues': ['equality', 'diversity', 'social', 'community'],
                    'Nostalgic': ['nostalgia', 'classic', 'memories', 'traditional']
                }
                
                themes = {}
                for theme, keywords in themes_keywords.items():
                    themes[theme] = any(keyword in description.lower() for keyword in keywords)
                
                return themes
            
            # Emotional tone analysis
            def analyze_emotional_tone(description: str) -> Dict[str, Any]:
                blob = TextBlob(description)
                
                # Sentiment analysis
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Categorize polarity
                def categorize_polarity(polarity):
                    if polarity > 0.5:
                        return 'Very Positive'
                    elif polarity > 0:
                        return 'Slightly Positive'
                    elif polarity == 0:
                        return 'Neutral'
                    elif polarity > -0.5:
                        return 'Slightly Negative'
                    else:
                        return 'Very Negative'
                
                return {
                    'polarity_score': polarity,
                    'subjectivity_score': subjectivity,
                    'emotional_category': categorize_polarity(polarity)
                }
            
            # Retrieve video transcript
            transcript = self.get_video_transcript(video_id)
            
            # Assemble video details
            video_details = {
                # Basic Information
                'video_id': video_id,
                'source_url': source_url,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'published_at': snippet.get('publishedAt', ''),
                'tags': snippet.get('tags', []),
                'default_language': snippet.get('defaultLanguage', ''),
                
                # Performance Metrics
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'engagement_rate': calculate_engagement_rate(
                    int(stats.get('viewCount', 0)), 
                    int(stats.get('likeCount', 0)), 
                    int(stats.get('commentCount', 0))
                ),
                
                # Content Details
                'duration': parse_duration(video_item.get('contentDetails', {}).get('duration', '')),
                'category_id': snippet.get('categoryId', ''),
                
                # Channel Information
                'channel_name': channel_info.get('snippet', {}).get('title', ''),
                'channel_subscriber_count': int(channel_info.get('statistics', {}).get('subscriberCount', 0)),
                
                # Ad-Specific Insights
                'potential_themes': extract_potential_themes(snippet.get('description', '')),
                'emotional_tone': analyze_emotional_tone(snippet.get('description', '')),
                
                # Transcript Information
                'transcript': transcript
            }
            
            # Save video details
            output_file = os.path.join(self.output_dir, 'details', f"{video_id}_details.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(video_details, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved details for video {video_id}")
            
            return video_details
        
        except Exception as e:
            logger.error(f"Error collecting details for video {video_id}: {e}")
            self.failed_videos.append({
                'youtube_id': video_id,
                'source_url': source_url,
                'reason': str(e)
            })
            return {}

    def collect_video_comments(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Collect and analyze video comments
        
        :param video_id: YouTube video ID
        :return: List of comment analyses
        """
        # Create robust session
        session = self._create_robust_session()
        
        try:
            comments_analysis = []
            base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            next_page_token = None
            comments_count = 0
            max_comments = 1000
            
            while comments_count < max_comments:
                params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'key': self.api_key,
                    'maxResults': 100,
                    'pageToken': next_page_token if next_page_token else ''
                }
                
                response = session.get(
                    base_url, 
                    params=params, 
                    headers=self._get_headers(),
                    timeout=10
                )
                data = response.json()
                
                for item in data.get('items', []):
                    comment_snippet = item['snippet']['topLevelComment']['snippet']
                    comment_text = comment_snippet.get('textDisplay', '')
                    
                    # Sentiment analysis
                    blob = TextBlob(comment_text)
                    
                    # Detailed comment analysis
                    comment_details = {
                        'video_id': video_id,
                        'comment_id': item['id'],
                        'text': comment_text,
                        'published_at': comment_snippet.get('publishedAt', ''),
                        'likes': comment_snippet.get('likeCount', 0),
                        'author': comment_snippet.get('authorDisplayName', ''),
                        
                        # Sentiment analysis
                        'sentiment_polarity': blob.sentiment.polarity,
                        'sentiment_subjectivity': blob.sentiment.subjectivity,
                        
                        # Sentiment category
                        'sentiment_category': (
                            'Strongly Positive' if blob.sentiment.polarity > 0.2 else
                            'Strongly Negative' if blob.sentiment.polarity < -0.2 else
                            'Neutral'
                        ),
                        
                        # Language features
                        'word_count': len(comment_text.split()),
                        'contains_emoji': bool(re.search(r'[^\w\s]', comment_text)),
                        
                        # Other metadata
                        'total_reply_count': item['snippet'].get('totalReplyCount', 0)
                    }
                    
                    comments_analysis.append(comment_details)
                    comments_count += 1
                    
                    if comments_count >= max_comments:
                        break
                
                # Pagination handling
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                
                # Random delay to mimic human behavior
                time.sleep(random.uniform(0.5, 1.5))
            
            # Save comments analysis as JSON
            comments_path = os.path.join(self.output_dir, 'comments', f"{video_id}_comments.json")
            with open(comments_path, 'w', encoding='utf-8') as f:
                json.dump(comments_analysis, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Collected {len(comments_analysis)} comments for video {video_id}")
            return comments_analysis
        
        except Exception as e:
            logger.error(f"Error collecting comments for video {video_id}: {e}")
            return []

    def analyze_batch(self, batch_size: int = 10):
        """
        Analyze videos in batches
        
        :param batch_size: Number of videos to process in each batch
        """
        # Reset failed videos list
        self.failed_videos = []
        
        # Process videos in batches
        for i in range(0, len(self.video_data), batch_size):
            batch = self.video_data[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} videos")
            
            # Process each video in the batch
            for video_item in batch:
                try:
                    # Get video details
                    video_details = self.get_youtube_video_details(video_item)
                    
                    # If video details retrieved successfully, get comments
                    if video_details:
                        self.collect_video_comments(video_details['video_id'])
                    
                    # Random delay between videos
                    time.sleep(random.uniform(1, 3))
                
                except Exception as e:
                    logger.error(f"Error processing video {video_item.get('youtube_id')}: {e}")
            
            # Longer pause between batches
            time.sleep(random.uniform(5, 10))
        
        # Log and save failed videos
        self._log_failed_videos()

    def _log_failed_videos(self):
        """
        Log and save information about failed video retrievals
        """
        if not self.failed_videos:
            logger.info("All videos processed successfully!")
            return
        
        # Prepare output paths
        json_path = os.path.join(self.output_dir, 'failed_videos.json')
        csv_path = os.path.join(self.output_dir, 'failed_videos.csv')
        
        # Save as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.failed_videos, f, ensure_ascii=False, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(self.failed_videos)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.warning(f"Failed to process {len(self.failed_videos)} videos. Details saved to {json_path} and {csv_path}")
        
        # Print failed video details
        for failed in self.failed_videos:
            logger.warning(f"Failed Video - ID: {failed.get('youtube_id')}, URL: {failed.get('source_url')}, Reason: {failed.get('reason')}")

def main():
    # Configuration parameters
    API_KEY = input("Enter YouTube Data API key: ")
    INPUT_FILE = r'S:\Documents\2025 - Purdue\AAAAAPURDUE\5 Capstone\SuperBowlProject\data\raw\superbowl_ads\Youtube_ID_Yearly\2020.json'
    OUTPUT_DIR = r'S:\Documents\2025 - Purdue\AAAAAPURDUE\5 Capstone\SuperBowlProject\data\raw\ad_features'
    
    # Initialize analyzer
    analyzer = YouTubeVideoAnalyzer(
        api_key=API_KEY, 
        input_file=INPUT_FILE, 
        output_dir=OUTPUT_DIR
    )
    
    # Analyze videos in batches
    analyzer.analyze_batch(batch_size=5)


if __name__ == '__main__':
    main()