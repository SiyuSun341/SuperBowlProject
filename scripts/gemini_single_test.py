import os
import base64
import json
import logging
import google.generativeai as genai
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import librosa
import numpy as np

# Load environment variables
load_dotenv()

# ========== Environment Validation ==========
def check_env_variable():
    """
    Validate the presence of Google API key in environment variables
    
    Returns:
        str: API key
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logging.error("Missing environment variable: GOOGLE_API_KEY")
        exit(1)
    return api_key

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gemini_video_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Configure API Key
api_key = check_env_variable()
genai.configure(api_key=api_key)

# ========== Utility Functions ==========
def encode_image(image_path):
    """
    Encode image to base64
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_transcript(transcript_path):
    """
    Read transcript file
    
    Args:
        transcript_path (str): Path to transcript file
    
    Returns:
        str: Transcript text
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading transcript: {e}")
        return ""

def truncate_text(text, max_length=8000):
    """
    Truncate text to specified max length
    
    Args:
        text (str): Input text
        max_length (int): Maximum text length
    
    Returns:
        str: Truncated text
    """
    return text[:max_length] + "\n...[truncated due to size limit]" if len(text) > max_length else text

def save_json_safe(data, path, max_length=1_000_000):
    """
    Safely save JSON data
    
    Args:
        data (dict): Data to save
        path (str): Output file path
        max_length (int): Maximum JSON length
    """
    try:
        data = json.dumps(data, indent=2, ensure_ascii=False)
        if len(data) > max_length:
            data = truncate_text(data, max_length)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")

# ========== Gemini Analysis Functions ==========
def analyze_video_frames(model, frame_dir, max_frames=5):
    """
    Analyze video frames
    
    Args:
        model: Gemini model
        frame_dir (str): Directory with frame images
        max_frames (int): Maximum frames to analyze
    
    Returns:
        list: Frame analyses
    """
    frame_analyses = []
    all_frames = [f for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_frames = random.sample(all_frames, min(max_frames, len(all_frames)))

    for frame_file in selected_frames:
        frame_path = os.path.join(frame_dir, frame_file)
        try:
            base64_image = encode_image(frame_path)
            prompt = """Perform a comprehensive multi-dimensional analysis of this image:

1. Visual Composition
- Color palette and mood
- Lighting and contrast
- Composition techniques
- Visual hierarchy
- Potential emotional impact

2. Content Analysis
- Detailed description of visual elements
- Identification of:
  * People (celebrities, actors, demographics)
  * Objects and props
  * Brands or logos
  * Background and setting

3. Advertising Potential
- Potential marketing messages
- Target audience interpretation
- Emotional triggers
- Visual storytelling elements
- Potential brand alignment

4. Semantic and Contextual Insights
- Symbolic or metaphorical elements
- Cultural references
- Potential narrative or story arc
- Unique visual metaphors

5. Technical Image Quality
- Resolution and clarity
- Potential professional production techniques
- Cinematography style
- Post-processing effects"""

            image_parts = [{"mime_type": "image/png", "data": base64_image}]
            response = model.generate_content(parts=[*image_parts, prompt])

            frame_analyses.append({
                "frame_name": frame_file,
                "analysis": truncate_text(response.text)
            })
            logging.info(f"Analyzed frame: {frame_file}")
        except Exception as e:
            logging.error(f"Frame analysis failed for {frame_file}: {e}")

    return frame_analyses

def analyze_audio_transcript(model, transcript_path):
    """
    Analyze audio transcript
    
    Args:
        model: Gemini model
        transcript_path (str): Path to transcript file
    
    Returns:
        dict: Transcript analysis
    """
    try:
        transcript = read_transcript(transcript_path)
        prompt = f"""Perform a comprehensive multi-dimensional analysis of the following transcript:

1. Linguistic Characteristics
- Tone and style
- Language complexity
- Rhetorical devices
- Emotional language markers

2. Content Structure
- Main narrative themes
- Storytelling techniques
- Key message or proposition
- Narrative arc

3. Communication Strategy
- Target audience communication style
- Persuasion techniques
- Emotional triggers
- Implicit and explicit messaging

4. Semantic Analysis
- Key phrases and their significance
- Metaphors and symbolic language
- Subtext and underlying meanings
- Cultural references

5. Advertising Insights
- Potential marketing messages
- Brand positioning
- Product or service positioning
- Call-to-action effectiveness

Transcript:\n{transcript}"""

        response = model.generate_content(parts=[prompt])
        return {"full_analysis": truncate_text(response.text)}
    except Exception as e:
        logging.error(f"Transcript analysis failed: {e}")
        return None

def analyze_audio_file(model, audio_path):
    """
    Analyze audio file's multiple dimensions
    
    Args:
        model: Gemini model
        audio_path (str): Path to audio file
    
    Returns:
        dict: Audio analysis results
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Extract audio basic features
        features = {
            "duration": librosa.get_duration(y=y, sr=sr),
            "sample_rate": sr,
            "rms_energy": np.mean(librosa.feature.rms(y=y)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=y)),
        }
        
        # Pitch and timbre features
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        features["mean_pitch"] = np.mean(pitch)
        
        # Audio emotional and atmosphere analysis prompt
        prompt = f"""Perform a multi-dimensional emotional and technical audio analysis:

1. Emotional Atmosphere Analysis
- Overall emotional tone
- Music/sound emotional temperature
- Potential psychological suggestions
- Emotional transition points

2. Technical Audio Features
- Audio duration: {features['duration']:.2f} seconds
- Average energy: {features['rms_energy']:.4f}
- Pitch characteristics: {features['mean_pitch']:.2f}
- Sound dynamic variation features

3. Advertising Audio Strategy
- Background music emotional guidance
- Sound design marketing strategy
- How audio reinforces ad message
- Target audience auditory experience

4. Sound Element Identification
- Background music style
- Voice-over characteristics
- Laughter, sound effects
- Sound layering

5. Narrative and Emotional Guidance
- How audio supports visual narrative
- Sound emotional guidance strategy
- Key emotional turning points"""
        
        # Gemini analysis
        response = model.generate_content(parts=[prompt])
        
        return {
            "technical_features": features,
            "audio_analysis": truncate_text(response.text)
        }
    
    except Exception as e:
        logging.error(f"Audio analysis failed: {e}")
        return None

# ========== Parallel Processing ==========
def process_single_video(youtube_id, video_path, analysis_output_dir, model):
    """
    Process a single video
    
    Args:
        youtube_id (str): YouTube video ID
        video_path (str): Path to video folder
        analysis_output_dir (str): Output directory for analysis
        model: Gemini model
    """
    try:
        frames_dir = os.path.join(video_path, "frames")
        transcript_path = os.path.join(video_path, f"{youtube_id}_transcript.txt")
        audio_path = os.path.join(video_path, "audio.mp3")
        
        # Analyze key frames
        frame_analyses = analyze_video_frames(model, frames_dir)
        
        # Analyze transcript
        transcript_analysis = analyze_audio_transcript(model, transcript_path)
        
        # Analyze audio file
        audio_analysis = analyze_audio_file(model, audio_path)

        # Comprehensive analysis
        video_analysis = {
            "video_id": youtube_id,
            "frame_analyses": frame_analyses,
            "transcript_analysis": transcript_analysis,
            "audio_analysis": audio_analysis
        }

        output_path = os.path.join(analysis_output_dir, f"{youtube_id}_gemini_video_analysis.json")
        save_json_safe(video_analysis, output_path)
        logging.info(f"✅ Saved analysis for {youtube_id}")
    
    except Exception as e:
        logging.error(f"❌ Failed analysis for {youtube_id}: {e}")

# ========== Main Workflow ==========
def comprehensive_video_analysis(processed_root, max_workers=1, max_videos=1):
    """
    Analyze videos with configurable concurrency and video count
    
    Args:
        processed_root (str): Root directory of processed videos
        max_workers (int): Maximum concurrent workers
        max_videos (int): Maximum number of videos to process
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    analysis_output_dir = os.path.join(processed_root, "comprehensive_video_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)

    # Get video folders, excluding analysis output folder
    video_folders = [
        folder for folder in os.listdir(processed_root) 
        if os.path.isdir(os.path.join(processed_root, folder)) 
        and folder != "comprehensive_video_analysis"
    ]

    # Limit processed videos
    video_folders = video_folders[:max_videos]

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for youtube_id in video_folders:
            video_path = os.path.join(processed_root, youtube_id)
            tasks.append(executor.submit(process_single_video, youtube_id, video_path, analysis_output_dir, model))

        for future in as_completed(tasks):
            future.result()

    logging.info(f"✅ Analysis completed for {len(tasks)} videos")

# ========== Run ==========
def main():
    processed_root = r"D:\youtube_vedio\2025\processed"
    
    # Process first video
    comprehensive_video_analysis(
        processed_root, 
        max_workers=1,  # Single thread
        max_videos=1    # First video only
    )

if __name__ == "__main__":
    main()

'''
# 处理第一个视频
comprehensive_video_analysis(processed_root, max_workers=1, max_videos=1)

# 处理前3个视频，2个并发
comprehensive_video_analysis(processed_root, max_workers=2, max_videos=3)

# 处理所有视频
comprehensive_video_analysis(processed_root, max_workers=4, max_videos=len(os.listdir(processed_root)))
'''