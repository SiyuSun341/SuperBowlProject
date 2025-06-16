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

api_key = check_env_variable()
genai.configure(api_key=api_key)

# ========== Utility Functions ==========
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_transcript(transcript_path):
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading transcript: {e}")
        return ""

def truncate_text(text, max_length=8000):
    return text[:max_length] + "\n...[truncated due to size limit]" if len(text) > max_length else text

def save_json_safe(data, path, max_length=1_000_000):
    try:
        data = json.dumps(data, indent=2, ensure_ascii=False)
        if len(data) > max_length:
            data = truncate_text(data, max_length)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")

def load_metadata(youtube_id):
    try:
        metadata_path = os.path.join("data", "raw", "ad_features", "details", f"{youtube_id}_details.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"No metadata found for {youtube_id}: {e}")
        return {}

def format_metadata_prompt(metadata):
    if not metadata:
        return ""
    return f"""Ad Metadata for reference:
Title: {metadata.get('title', '')}
Description: {metadata.get('description', '')}
Published At: {metadata.get('published_at', '')}
Channel: {metadata.get('channel_name', '')}
View Count: {metadata.get('view_count', '')}
Like Count: {metadata.get('like_count', '')}
Engagement Rate: {metadata.get('engagement_rate', '')}
Emotional Tone: Polarity {metadata.get('emotional_tone', {}).get('polarity_score', '')}, Subjectivity {metadata.get('emotional_tone', {}).get('subjectivity_score', '')}, Category {metadata.get('emotional_tone', {}).get('emotional_category', '')}
Tags: {', '.join(metadata.get('tags', []))}
"""

# ========== Gemini Analysis ==========
def analyze_video_frames(model, frame_dir, metadata_prompt, max_frames=3):
    frame_analyses = []
    all_frames = [f for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_frames = random.sample(all_frames, min(max_frames, len(all_frames)))

    for frame_file in selected_frames:
        frame_path = os.path.join(frame_dir, frame_file)
        try:
            base64_image = encode_image(frame_path)
            prompt = metadata_prompt + """
Analyze this image:
1. Visual Style: mood, color, lighting, composition.
2. Content: people, objects, logos, background.
3. Marketing Clues: key messages, emotion, target appeal.
4. Symbolism: metaphors, culture, story hints.
5. Technical: image clarity, camera work, editing hints.
"""
            response = model.generate_content([
                {"mime_type": "image/png", "data": base64_image},
                prompt
            ])
            frame_analyses.append({
                "frame_name": frame_file,
                "analysis": truncate_text(response.text)
            })
            logging.info(f"Analyzed frame: {frame_file}")
        except Exception as e:
            logging.error(f"Frame analysis failed for {frame_file}: {e}")

    return frame_analyses

def analyze_audio_transcript(model, transcript_path, metadata_prompt):
    try:
        transcript = read_transcript(transcript_path)
        prompt = metadata_prompt + f"""
Analyze this ad transcript:
1. Tone: language style, emotion, complexity.
2. Structure: key messages, story arc, narration.
3. Strategy: audience fit, persuasion, emotional pulls.
4. Meaning: key phrases, metaphors, hidden subtext.
5. Ad Potential: brand voice, positioning, CTA strength.

Transcript:
{transcript}
"""
        response = model.generate_content(prompt)
        return {"full_analysis": truncate_text(response.text)}
    except Exception as e:
        logging.error(f"Transcript analysis failed: {e}")
        return None

def analyze_audio_file(model, audio_path, metadata_prompt):
    try:
        y, sr = librosa.load(audio_path)
        features = {
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "sample_rate": int(sr),
            "rms_energy": float(np.mean(librosa.feature.rms(y=y))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        }
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        features["mean_pitch"] = float(np.mean(pitch))

        prompt = metadata_prompt + """
Analyze the audio track:
1. Emotion: mood, music tone, emotional cues.
2. Audio Metrics: duration, energy, pitch, clarity.
3. Ad Use: how music or voice supports messaging.
4. Elements: music style, voiceover, effects, layers.
5. Narrative: how sound builds emotion or plot.
"""
        response = model.generate_content(prompt)
        return {
            "technical_features": features,
            "audio_analysis": truncate_text(response.text)
        }
    except Exception as e:
        logging.error(f"Audio analysis failed: {e}")
        return None

# ========== Parallel Processing ==========
# ========== Parallel Processing ==========
def process_single_video(youtube_id, video_path, analysis_output_dir, model):
    # 预定义默认结构
    video_analysis = {
        "video_id": youtube_id,
        "frame_analyses": [],
        "transcript_analysis": {
            "full_analysis": "No transcript analysis available"
        },
        "audio_analysis": {
            "technical_features": {
                "duration": 0.0,
                "sample_rate": 0,
                "rms_energy": 0.0,
                "zero_crossing_rate": 0.0,
                "mean_pitch": 0.0
            },
            "audio_analysis": "No audio analysis available"
        }
    }

    try:
        frames_dir = os.path.join(video_path, "frames")
        transcript_path = os.path.join(video_path, f"{youtube_id}_transcript.txt")
        audio_path = os.path.join(video_path, "audio.mp3")
        metadata = load_metadata(youtube_id)
        metadata_prompt = format_metadata_prompt(metadata)

        # 帧分析
        if os.path.exists(frames_dir) and os.listdir(frames_dir):
            frame_analyses = analyze_video_frames(model, frames_dir, metadata_prompt)
            if frame_analyses:
                video_analysis["frame_analyses"] = frame_analyses

        # 转录分析
        if os.path.exists(transcript_path):
            transcript_analysis = analyze_audio_transcript(model, transcript_path, metadata_prompt)
            if transcript_analysis:
                video_analysis["transcript_analysis"] = transcript_analysis

        # 音频分析
        if os.path.exists(audio_path):
            audio_analysis = analyze_audio_file(model, audio_path, metadata_prompt)
            if audio_analysis:
                video_analysis["audio_analysis"] = audio_analysis

        # 保存分析结果
        output_path = os.path.join(analysis_output_dir, f"{youtube_id}_gemini_video_analysis.json")
        save_json_safe(video_analysis, output_path)
        logging.info(f"✅ Saved analysis for {youtube_id}")

    except Exception as e:
        # 如果出现任何错误，仍然保存默认结构
        output_path = os.path.join(analysis_output_dir, f"{youtube_id}_gemini_video_analysis.json")
        save_json_safe(video_analysis, output_path)
        logging.error(f"❌ Partial analysis for {youtube_id}: {e}")

    return video_analysis

# ========== Main Workflow ==========
def comprehensive_video_analysis(processed_root, max_workers=1, max_videos=1):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    analysis_output_dir = os.path.join(processed_root, "comprehensive_video_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)

    video_folders = [
        folder for folder in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, folder))
        and folder != "comprehensive_video_analysis"
    ]
    video_folders = video_folders[:max_videos]

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for youtube_id in video_folders:
            video_path = os.path.join(processed_root, youtube_id)
            tasks.append(executor.submit(process_single_video, youtube_id, video_path, analysis_output_dir, model))

        for future in as_completed(tasks):
            future.result()

    logging.info(f"✅ Analysis completed for {len(tasks)} videos")

def main():
    processed_root = r"D:\\youtube_vedio\\2020\\processed"
    comprehensive_video_analysis(processed_root, max_workers=5, max_videos=len(os.listdir(processed_root)))

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