import os
import subprocess
import json
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def extract_frames(video_path, frames_dir):
    """Extract video key frames"""
    try:
        subprocess.run([
            "ffmpeg", 
            "-i", video_path,
            "-vf", "fps=1/5",  # 1 frame per 5 seconds, adjustable
            "-q:v", "2",  # High quality
            os.path.join(frames_dir, "frame_%04d.png")
        ], check=True)
        logging.info(f"Frames extracted successfully from {video_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Frame extraction failed for {video_path}: {e}")
        raise

def extract_audio(video_path, audio_path):
    """Extract audio"""
    try:
        # Check if audio stream exists
        audio_tracks = subprocess.check_output([
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "a", 
            "-show_entries", "stream=index", 
            "-of", "csv=p=0", 
            video_path
        ]).decode().strip().split("\n")
        
        if audio_tracks and audio_tracks[0]:
            subprocess.run([
                "ffmpeg", 
                "-i", video_path, 
                "-vn",  # Audio only
                "-acodec", "libmp3lame",  # MP3 encoding
                "-b:a", "128k",  # Audio bitrate
                audio_path
            ], check=True)
            logging.info(f"Audio extracted successfully from {video_path}")
        else:
            logging.warning(f"No audio tracks found in {video_path}")
    except Exception as e:
        logging.error(f"Audio extraction failed for {video_path}: {traceback.format_exc()}")
        raise

def extract_subtitles(video_path, subtitle_path):
    """Extract subtitles"""
    try:
        # Check if subtitle stream exists
        subtitle_tracks = subprocess.check_output([
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "s", 
            "-show_entries", "stream=index", 
            "-of", "csv=p=0", 
            video_path
        ]).decode().strip().split("\n")
        
        if subtitle_tracks and subtitle_tracks[0]:
            subprocess.run([
                "ffmpeg", 
                "-i", video_path, 
                "-map", f"0:s:{subtitle_tracks[0]}", 
                subtitle_path
            ], check=True)
            logging.info(f"Subtitles extracted successfully from {video_path}")
        else:
            logging.warning(f"No subtitle tracks found in {video_path}")
    except Exception as e:
        logging.warning(f"Subtitle extraction failed for {video_path}: {e}")

def find_video_directory():
    """
    Find a suitable video input directory
    
    Returns:
        str: Path to video directory
    """
    possible_paths = [
        'videos',
        'input_videos',
        os.path.join('data', 'videos'),
        os.path.join('data', 'raw', 'videos')
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    # Create a default directory if none exists
    os.makedirs('videos', exist_ok=True)
    return 'videos'

def process_video(video_root, output_root):
    """Process video files"""
    os.makedirs(output_root, exist_ok=True)
    
    # Find all video files
    video_files = [
        f for f in os.listdir(video_root) 
        if f.lower().endswith(('.mp4', '.mkv', '.webm', '.avi'))
    ]
    
    process_log = {}

    for video_file in video_files:
        try:
            # Video ID and path
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_root, video_file)
            
            # Create output directories
            output_dir = os.path.join(output_root, video_id)
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # File paths
            audio_path = os.path.join(output_dir, "audio.mp3")
            subtitle_path = os.path.join(output_dir, "subtitles.srt")
            
            # Log entry
            log_entry = {
                "video": video_file, 
                "status": "success", 
                "errors": []
            }
            
            # Extract frames
            extract_frames(video_path, frames_dir)
            
            # Extract audio
            extract_audio(video_path, audio_path)
            
            # Extract subtitles
            extract_subtitles(video_path, subtitle_path)
            
            process_log[video_id] = log_entry
            
        except Exception as e:
            log_entry = {
                "video": video_file, 
                "status": "failed", 
                "errors": [str(e)]
            }
            process_log[video_id] = log_entry
            logging.error(f"Processing failed for {video_file}: {traceback.format_exc()}")

    # Save log
    log_path = os.path.join(output_root, "processing_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(process_log, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ All processing completed. Log saved to: {log_path}")

def main():
    # Find input and output directories dynamically
    video_root = find_video_directory()
    output_root = os.path.join('processed_videos')
    os.makedirs(output_root, exist_ok=True)
    
    process_video(video_root, output_root)

if __name__ == "__main__":
    main()