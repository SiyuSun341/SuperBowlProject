# Super Bowl Advertisement Analysis for Rogue Ridge

## Project Overview

A comprehensive data-driven analysis of Super Bowl commercials to develop a strategic advertising approach for Forge & Field's new Rogue Ridge personal care product line.

### Business Context

* **Client**: Forge & Field Brands
* **Product**: Rogue Ridge (Personal Care Line for American Men)
* **Investment**: \$11M total advertising budget
* **Key Objective**: Develop a high-impact 30-second Super Bowl commercial

---

## Project Status

![Data Collection](https://img.shields.io/badge/Data%20Collection-Completed-green)
![Model Training](https://img.shields.io/badge/Model%20Training-In%20Progress-yellow)
![Analysis](https://img.shields.io/badge/Preliminary%20Analysis-Completed-green)

**Last Updated**: June 2025
**Current Phase**: Deliverable 2 - Model Training and Preliminary Insights

---

## Data Collection Overview

### Data Sources and Volume

* **YouTube**: 1,181 Super Bowl ad videos (2000-2025)

  * 470 with metadata and comments (\~50,000 comments)
* **Reddit**: \~10,000 posts/comments (2020-2024)
* **News Articles**: \~500 articles linked from Reddit
* **Video Files**: 1,181 downloaded MP4s
* **Multimodal Content per Ad**:

  * Audio (MP3), Subtitle (TXT), Keyframes (JPG)

### Data Completeness

* **Video Metadata**: >95% complete
* **Reddit & News**: Supplemented for missing YouTube discussions
* **Comment Richness**: Multi-source, sentiment-scored

### Key Scripts

* `extract_youtube_id_list.py`: Scrape YouTube IDs from superbowl-ads.com
* `youtube_info.py`: Fetch metadata & comments
* `reddit_updata.py`: Search Reddit discussions using PRAW
* `superbowldownload.py`: MP4 download fallback for non-YouTube videos
* `whisper_audio_process.py`: Transcribe audio using Whisper

---

## AI/ML Workflow Summary

### Step-by-Step Pipeline

1. **Preprocessing**:

   * Clean comments, subtitles, descriptions
   * Remove emojis, filler text, duplicates

2. **Sentiment Classification**:

   * Models: TextBlob, VADER, RoBERTa, FinBERT, BART
   * Output: Positive / Neutral / Negative (via majority vote)

3. **Multimodal Feature Extraction**:

   * Tools: Whisper (audio), FFmpeg (video), Gemini API
   * Extracted: Mood, Emotion, Pacing, Slogan, Tone, Symbols

4. **Feature Engineering**:

   * LabelEncoder for categorical features
   * StandardScaler for numerical values
   * PCA to 20 components

5. **Model Training**:

   * Models: Logistic Regression, Random Forest, SVM, NB, KNN, MLP, CatBoost
   * Best Model: **CatBoost** with **87.3% test accuracy**

6. **Validation**:

   * GridSearchCV + 3-fold CV
   * EarlyStopping & max\_depth to prevent overfitting
   * Feature importance visualization


---

## Gemini Output Feature Tags

### Categories Extracted:

* **Visual Style**: Color\_Tone, Lighting, Composition, Style\_Tag
* **Mood/Narrative**: Emotional\_Tone, Structure, Pacing, Twist
* **Semantics**: Setting, Product Visibility, Masculine Symbols
* **Audio/Text**: Slogan, Narration Style, Humor Use
* **Audience Profile**: Gender, Age, Culture, Lifestyle


---

## Next Steps

| Task                       | Status | Action                                                     |
| -------------------------- | ------ | ---------------------------------------------------------- |
| Add 2025 Reddit data       | ⏳      | Use `reddit_updata.py` with year filtering                 |
| Multi-label classification | ✅      | Implement sentiment scoring vector                         |
| Prompt Engineering         | ✅      | Refine Gemini instructions for clarity                     |
| Resonance Modeling         | ❌      | Design alignment test between ad tone and audience segment |

---

## Repository Structure

```
superbowlproject/
├── config/                  # API keys, prompt templates
├── database/                # Processed data & backups
├── deliverable-2-appendix/ # Attachments and examples
├── logs/                   # Logging outputs
├── models/                 # ML models and scripts
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                
├── src/                    # Core modules (scraping, processing, analysis)
├── tests/                  # Unit & integration tests
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## Installation & Quick Start

### Requirements

* Python 3.8+
* Reddit API credentials
* YouTube Data API key
* Google Cloud (for Gemini)

### Setup

```bash
git clone https://github.com/SiyuSun341/SuperBowlProject.git
cd SuperBowlProject
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Fill in your API keys
```

### Run Example

```bash
python scripts/run_data_collection.py
python scripts/train_models.py
python scripts/generate_insights.py
```

---

## Documentation

| File                            | Description                 |
| -----------------------------   | --------------------------- |
| `docs/api_documentation.md`     | API usage reference         |
| `docs/model_documentation.md`   | ML model descriptions       |
| `docs/setup_guide.md`           | Environment & dependencies  |
| `docs/user_guide.md`            | Analysis walkthrough        |
| `docs/superbowl_python_setup.md`| superbowl setup             |


---

## Success Metrics

### Data

* 1181 total ads (1205 total vedios)
* > 95% completeness
* > 500K total comments

### Models

* CatBoost: 87.3% test accuracy
* PCA-d: 20 features, interpretable
* Cross-validation: stable across folds

### Strategic Insights
* Precisely identify 5-7 key success factors for Super Bowl advertisements
* Tailored advertising strategy for the Rogue Ridge personal care product line
* Provide data-driven advertising design recommendations

### Business Value
* Strategic decision support for $11 million advertising investment
* Actionable creative guidance for a 30-second commercial
* Mitigate commercial failure risks
* Help Forge & Field precisely target the intended audience (prototypical American men)

### Key Deliverables
* Comprehensive technical analysis report
* Client-facing advertising design recommendations document
* Pre-launch advertisement testing and optimization framework


---

## Contact

**Author**: Siyu Sun
**Email**: [sunsiyu.suzy@gmail.com](mailto:sunsiyu.suzy@gmail.com)
**GitHub**: [SiyuSun341](https://github.com/SiyuSun341/SuperBowlProject)

---

## License

This repository is part of an academic project at Purdue University (MBT Program). Do not distribute without permission.


