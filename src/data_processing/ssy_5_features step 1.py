import os
import json
import pandas as pd
from glob import glob
from collections import Counter
from textblob import TextBlob
import re

# === file path setting ===
json_dir = r".\video_gemini_analysis"
json_files = glob(os.path.join(json_dir, "*.json"))

# === auxiliary functions ===
def extract_nlp_features(text, label):
    pattern = rf"{re.escape(label)}\s*[:：]?\s*\*?\*?\s*(.*?)\n"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_entities(text):
    text = text.lower()
    people = re.findall(r"\b(man|woman|child|driver|person|people|actor|dj khaled|j\.?lo)\b", text)
    objects = re.findall(r"\b(car|gun|phone|table|logo|vehicle|steering wheel|pringle|can|microphone|spacesuit|patch|bag|rope|chandelier|mustache|bling cup)\b", text)
    places = re.findall(r"\b(room|street|hall|sky|background|scene|landscape|field|space|ballroom|cliff|cityscape|casino|spa|hotel|lobby)\b", text)
    return ', '.join(set(people)), ', '.join(set(objects)), ', '.join(set(places))

def most_common(lst):
    lst = [x for x in lst if x]
    return Counter(lst).most_common(1)[0][0] if lst else None

def extract_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def extract_by_keywords(text, keywords):
    for kw in keywords:
        match = extract_nlp_features(text, kw)
        if match:
            return match
    return None

# === main function ===
def extract_video_features(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_id = data.get("video_id", os.path.basename(json_path))
    moods, colors, lightings, compositions = [], [], [], []
    emotions, marketing_msgs, appeals = [], [], []
    metaphors, technical_edits = 0, []
    people_list, object_list, place_list = [], [], []

    for frame in data.get("frame_analyses", []):
        analysis = frame.get("analysis", "")
        moods.append(extract_nlp_features(analysis, "Mood"))
        colors.append(extract_nlp_features(analysis, "Color"))
        lightings.append(extract_nlp_features(analysis, "Lighting"))
        compositions.append(extract_nlp_features(analysis, "Composition"))
        emotions.append(extract_nlp_features(analysis, "Emotion"))
        marketing_msgs.append(extract_by_keywords(analysis, ["Key Message"]))
        appeals.append(extract_by_keywords(analysis, ["Target Appeal"]))
        if "metaphor" in analysis.lower() or "symbol" in analysis.lower():
            metaphors += 1
        technical_edits.append(extract_by_keywords(analysis, ["Editing Hints"]))
        people, objects, places = extract_entities(analysis)
        people_list.append(people)
        object_list.append(objects)
        place_list.append(places)

    # text analysis fields
    transcript = data.get("transcript_analysis", {}).get("full_analysis", "")
    polarity, subjectivity = extract_sentiment(transcript)
    transcript_style = extract_by_keywords(transcript, ["Language Style"])
    transcript_emotion = extract_by_keywords(transcript, ["Emotion"])
    transcript_audience = extract_by_keywords(transcript, ["Audience Fit"])
    transcript_keyphrase = extract_by_keywords(transcript, ["Key Phrases"])

    # audio technical features
    audio_meta = data.get("audio_analysis", {}).get("technical_features", {})
    pitch = audio_meta.get("mean_pitch")
    energy = audio_meta.get("rms_energy")
    zcr = audio_meta.get("zero_crossing_rate")
    duration = audio_meta.get("duration")
    sample_rate = audio_meta.get("sample_rate")

    return {
        "video_id": video_id,
        "image_mood": most_common(moods),
        "image_color": most_common(colors),
        "image_lighting": most_common(lightings),
        "image_composition": most_common(compositions),
        "image_emotion": most_common(emotions),
        "image_marketing_message": most_common(marketing_msgs),
        "image_target_appeal": most_common(appeals),
        "image_metaphor_count": metaphors,
        "image_technical_editing": most_common(technical_edits),
        "image_people_keywords": most_common(people_list),
        "image_object_keywords": most_common(object_list),
        "image_place_keywords": most_common(place_list),
        "transcript_polarity": polarity,
        "transcript_subjectivity": subjectivity,
        "transcript_style": transcript_style,
        "transcript_emotion": transcript_emotion,
        "transcript_audience": transcript_audience,
        "transcript_keyphrase": transcript_keyphrase,
        "audio_mean_pitch": pitch,
        "audio_rms_energy": energy,
        "audio_zero_crossing_rate": zcr,
        "audio_duration": duration,
        "audio_sample_rate": sample_rate
    }

# === 主流程 ===
if __name__ == "__main__":
    records = []
    for path in json_files:
        try:
            rec = extract_video_features(path)
            records.append(rec)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    df = pd.DataFrame(records)
    output_path = os.path.join(json_dir, "video_feature_analysis_full.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Features saved to {output_path}")
