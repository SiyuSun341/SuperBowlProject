import os
import whisper
import logging
import torch
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('whisper_large_transcription.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def format_timestamp(seconds):
    """
    Format timestamp to SRT standard format
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio(audio_path, model):
    """
    Transcribe a single audio file, optimized for large model
    
    Args:
        audio_path (str): Path to audio file
        model: Whisper model
    
    Returns:
        dict: Transcription result
    """
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Start timing
        start_time = time.time()

        # Perform speech transcription with large model optimization
        result = model.transcribe(
            audio_path, 
            language='en',
            fp16=torch.cuda.is_available(),  # Use half-precision only on GPU
            beam_size=5,  # Increase beam search size for accuracy
            patience=1.0,  # Control beam search patience
            condition_on_previous_text=False,  # Reduce context dependency
            compression_ratio_threshold=2.4,  # Control transcription quality
            no_speech_threshold=0.6  # Filter low-confidence segments
        )

        # End timing
        end_time = time.time()
        logging.info(f"Transcription time: {end_time - start_time:.2f} seconds")

        return result
    except Exception as e:
        logging.error(f"Transcription failed for {audio_path}: {e}")
        return None

def save_transcription(youtube_id, result, processed_dir):
    """
    Save transcription results
    
    Args:
        youtube_id (str): YouTube video ID
        result (dict): Transcription result
        processed_dir (str): Processed folder path
    """
    if not result:
        return

    try:
        # Full text file
        txt_path = os.path.join(processed_dir, f"{youtube_id}_transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logging.info(f"Full text saved: {txt_path}")

        # SRT subtitle file
        srt_path = os.path.join(processed_dir, f"{youtube_id}_transcript.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text'].strip()

                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        logging.info(f"SRT subtitles saved: {srt_path}")

    except Exception as e:
        logging.error(f"Failed to save transcription results for {youtube_id}: {e}")

def batch_transcribe(processed_root, model_size="medium"):
    """
    Batch process audio files
    
    Args:
        processed_root (str): Root directory of processed videos
        model_size (str): Whisper model size
    """
    # Hardware check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load model
    logging.info(f"Loading {model_size} model")
    model = whisper.load_model(model_size).to(device)
    
    # Processing statistics
    total_processed = 0
    successful_transcriptions = 0
    start_total_time = time.time()

    # Iterate through directory
    for youtube_id in os.listdir(processed_root):
        video_path = os.path.join(processed_root, youtube_id)
        
        # Ensure it's a directory
        if os.path.isdir(video_path):
            audio_path = os.path.join(video_path, "audio.mp3")
            
            # Check if audio file exists
            if os.path.exists(audio_path):
                total_processed += 1
                logging.info(f"Processing audio: {audio_path}")
                
                # Transcribe
                result = transcribe_audio(audio_path, model)
                
                # Save
                if result:
                    save_transcription(youtube_id, result, video_path)
                    successful_transcriptions += 1
    
    # Record processing summary
    end_total_time = time.time()
    logging.info(f"Processing completed - Total: {total_processed}, Successful: {successful_transcriptions}")
    logging.info(f"Total processing time: {end_total_time - start_total_time:.2f} seconds")

def main():
    # Processing directory
    processed_root = r"D:\youtube_vedio\2020\processed"
    
    # Batch transcription
    batch_transcribe(processed_root, model_size="medium")

if __name__ == "__main__":
    main()