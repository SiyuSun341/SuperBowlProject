from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import os

def get_youtube_service(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def search_videos(youtube, query, max_results=20):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    return response['items']

def get_video_comments(youtube, video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        while request and len(comments) < 1000:  # Limit to 1000 comments per video
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'video_id': video_id,
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })
            
            # Get the next page of comments
            request = youtube.commentThreads().list_next(request, response)
    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {str(e)}")
    
    return comments

def main():
<<<<<<< HEAD
    API_KEY = ""
=======
    API_KEY = "your api key"
>>>>>>> 72087d38e23fdea35548f560e950e62eab3d0d58
    search_query = input("Enter your search query (e.g., '2025 super bowl ad'): ")
    
    # Create YouTube API service
    youtube = get_youtube_service(API_KEY)
    
    # Search for videos
    print(f"Searching for videos with query: {search_query}")
    videos = search_videos(youtube, search_query)
    
    all_comments = []
    
    # Get comments for each video
    for i, video in enumerate(videos, 1):
        video_id = video['id']['videoId']
        video_title = video['snippet']['title']
        print(f"\nProcessing video {i}/20: {video_title}")
        
        comments = get_video_comments(youtube, video_id)
        all_comments.extend(comments)
        print(f"Found {len(comments)} comments")
    
    # Create DataFrame and save to CSV
    if all_comments:
        df = pd.DataFrame(all_comments)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"youtube_comments_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nComments saved to {filename}")
    else:
        print("No comments were found.")

if __name__ == "__main__":
    main() 
