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

<<<<<<< HEAD
```bash
python scripts/run_data_collection.py
python scripts/train_models.py
python scripts/generate_insights.py
=======
## Project Structure

```
superbowl-ad-analysis/├── README.md                          # Project overview and usage instructions
├── .gitignore                         # Git ignore file configuration
├── requirements.txt                   # Python dependencies list
├── environment.yml                    # Conda environment configuration (optional)
├── setup.py                          # Project installation configuration
├── LICENSE                           # Open source license
│
├── config/                           # Configuration files directory
│   ├── api_keys.yaml.template        # API keys template
│   ├── database_config.yaml          # Database configuration
│   ├── model_config.yaml             # Model parameters configuration
│   └── logging_config.yaml           # Logging configuration
│
├── data/                             # Data directory
│   ├── raw/                          # Raw data
│   │   ├── youtube/                  # YouTube data
│   │   │   ├── comments/             # Comment data
│   │   │   ├── metadata/             # Video metadata
│   │   │   └── video_features/       # Video features
│   │   ├── reddit/                   # Reddit data
│   │   │   ├── posts/                # Post data
│   │   │   └── comments/             # Comment data
│   │   ├── news/                     # News article data
│   │   │   └── articles/             # News article content
│   │   ├── ad_features/              # Advertisement features data
│   │   │   └── ad_list.csv           # Advertisement list
│   │   └── superbowl_ads/            # Super Bowl advertisement data
│   │       ├── ad_list.csv           # Advertisement list
│   │       └── ad_features.csv       # Advertisement features data
│   │
│   ├── processed/                    # Processed data
│   │   ├── cleaned_comments.csv      # Cleaned comment data
│   │   ├── sentiment_scores.csv      # Sentiment analysis scores
│   │   ├── reddit_analysis.csv       # Reddit analysis results
│   │   └── combined_dataset.csv     # Final merged dataset
│   ├── external/                     # External data sources
│   │   ├── industry_reports/         # Industry reports
│   │   └── benchmark_data/           # Benchmark data
│   └── sample/                       # Sample data (to be submitted to Git)
│       ├── sample_comments.csv       # Sample comment data
│       ├── sample_reddit.csv         # Sample Reddit data
│       └── sample_features.csv       # Sample advertisement features data
│
├── src/                              # Source code directory
│   ├── __init__.py
│   ├── data_collection/              # Data collection module
│   │   ├── __init__.py
│   │   ├── youtube_scraper.py        # YouTube data collection script
│   │   ├── reddit_scraper.py         # Reddit data collection script
│   │   ├── news_scraper.py           # News data collection script
│   │   ├── gemini_analyzer.py        # Gemini API video analysis
│   │   └── utils/                    # Utility functions
│   │       ├── __init__.py
│   │       ├── api_helpers.py        # API helper functions
│   │       ├── rate_limiter.py       # Rate limit handling
│   │       └── data_validator.py     # Data validation tools
│   │
│   ├── data_processing/              # Data processing module
│   │   ├── __init__.py
│   │   ├── text_cleaner.py           # Text cleaning (noise removal, punctuation, etc.)
│   │   ├── sentiment_analyzer.py     # Sentiment analysis (e.g., VADER)
│   │   ├── feature_extractor.py      # Feature extraction (from raw data)
│   │   ├── data_merger.py            # Data merging (merging different data sources)
│   │   └── preprocessor.py           # Data preprocessing pipeline
│   │
│   ├── models/                       # Machine learning models
│   │   ├── __init__.py
│   │   ├── base_model.py             # Base model class
│   │   ├── sentiment_model.py        # Sentiment analysis model
│   │   ├── success_predictor.py      # Advertisement success prediction model
│   │   ├── clustering_model.py       # Clustering analysis model
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── metrics.py            # Model evaluation metrics
│   │       └── cross_validation.py   # Cross-validation
│   │
│   ├── analysis/                     # Analysis module
│   │   ├── __init__.py
│   │   ├── statistical_analysis.py   # Statistical analysis (e.g., descriptive statistics)
│   │   ├── trend_analysis.py         # Trend analysis
│   │   ├── correlation_analysis.py   # Correlation analysis
│   │   └── insight_generator.py      # Insight generation
│   │
│   ├── visualization/                # Visualization module
│   │   ├── __init__.py
│   │   ├── plots.py                  # Basic plotting and charts
│   │   ├── interactive_viz.py        # Interactive visualizations
│   │   ├── dashboard.py              # Dashboard generation
│   │   └── report_charts.py          # Charts for reports
│   │
│   └── utils/                        # General utility module
│       ├── __init__.py
│       ├── database.py               # Database operations (e.g., SQLite)
│       ├── file_handler.py           # File handling
│       ├── logger.py                 # Logging
│       ├── config_loader.py          # Configuration file loading
│       └── helpers.py                # Helper functions
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data exploration and cleaning
│   ├── 02_data_collection_test.ipynb # Data collection testing
│   ├── 03_preprocessing.ipynb        # Data preprocessing
│   ├── 04_model_training.ipynb       # Model training and evaluation
│   ├── 05_analysis.ipynb             # Analysis and visualization
│   ├── 06_insights_generation.ipynb  # Insight generation
│   └── scratch/                      # Temporary test notebooks
│       └── .gitkeep
│
├── scripts/                          # Executable scripts
│   ├── setup_environment.py          # Environment setup (install dependencies, configure)
│   ├── download_data.py              # One-click data download
│   ├── run_data_collection.py        # Run data collection (calls data scraping scripts)
│   ├── train_models.py               # Train machine learning models
│   ├── generate_reports.py           # Generate reports
│   └── deploy_dashboard.py           # Deploy dashboard
│
├── tests/                            # Test files
│   ├── __init__.py
│   ├── test_data_collection/
│   │   ├── test_youtube_scraper.py   # Test YouTube data scraper
│   │   ├── test_reddit_scraper.py    # Test Reddit data scraper
│   │   └── test_api_helpers.py       # Test API helper functions
│   ├── test_data_processing/
│   │   ├── test_text_cleaner.py      # Test text cleaning
│   │   ├── test_sentiment_analyzer.py# Test sentiment analysis
│   │   └── test_preprocessor.py      # Test data preprocessing pipeline
│   ├── test_models/
│   │   ├── test_base_model.py        # Test base model
│   │   └── test_success_predictor.py # Test advertisement success prediction model
│   └── test_utils/
│       ├── test_database.py          # Test database operations
│       └── test_helpers.py           # Test helper functions
│
├── docs/                             # Documentation directory
│   ├── README.md                     # Project overview and user guide
│   ├── api_documentation.md          # API documentation
│   ├── data_dictionary.md            # Data dictionary (description of each field)
│   ├── model_documentation.md        # Model documentation
│   ├── setup_guide.md                # Environment setup guide
│   └── user_guide.md                 # User guide
│
├── reports/                          # Reports and deliverables
│   ├── deliverables/
│   │   ├── deliverable_1.pdf         # Project plan
│   │   ├── deliverable_2.pdf         # Development status report
│   │   └── deliverable_3.pdf         # Final report
│   ├── technical_reports/
│   │   ├── data_collection_report.md
│   │   ├── model_performance_report.md
│   │   └── analysis_results.md
│   ├── client_reports/
│   │   ├── executive_summary.pdf
│   │   ├── detailed_findings.pdf
│   │   └── recommendations.pdf
│   └── presentations/
│       ├── project_overview.pptx
│       ├── methodology.pptx
│       └── final_presentation.pptx
│
├── models/                           # Trained model files
│   ├── sentiment_analysis/
│   │   ├── model.pkl
│   │   ├── tokenizer.pkl
│   │   └── config.json
│   ├── success_prediction/
│   │   ├── classifier.pkl
│   │   ├── features.pkl
│   │   └── scaler.pkl
│   └── clustering/
│       ├── kmeans_model.pkl
│       └── cluster_labels.csv
│
├── logs/                             # Log files (add to .gitignore)
│   ├── data_collection.log
│   ├── model_training.log
│   ├── analysis.log
│   └── error.log
│
├── database/                         # Database files (add to .gitignore)
│   ├── superbowl_ads.db             # SQLite database
│   └── backups/
│       └── .gitkeep
│
└── assets/                           # Static resources
    ├── images/
    │   ├── logos/
    │   ├── charts/
    │   └── screenshots/
    ├── templates/
    │   ├── report_template.html
    │   └── email_template.html
    └── style/
        ├── report_style.css
        └── dashboard_theme.css

```

---

## Data Sources

### Primary Sources
- **YouTube**: 500+ Super Bowl ads (2000-2025) with comments and metadata
- **Reddit**: Discussions from r/advertising, r/SuperBowl, r/marketing
- **News Media**: Rankings and reviews from AdAge, Marketing Land, USA Today
- **Gemini API**: Video content analysis and feature extraction

### Expected Data Volume
- **500** Super Bowl advertisement videos
- **500K+** YouTube comments
- **50K+** Reddit comments and posts
- **1000+** news articles and reviews
- **20+** features per advertisement

---

## AI/ML Pipeline

### 1. Data Collection & Processing
```python
# Automated data collection
python scripts/run_data_collection.py

# Data preprocessing pipeline
from src.data_processing import preprocessor
pipeline = preprocessor.create_pipeline()
cleaned_data = pipeline.transform(raw_data)
```

### 2. Feature Engineering
- **Text Features**: TF-IDF, n-grams, topic modeling
- **Sentiment Analysis**: VADER, TextBlob, RoBERTa
- **Video Features**: Humor, celebrities, animals, music type
- **Engagement Metrics**: Likes, shares, comments ratios

### 3. Model Training
```python
# Train success prediction model
from src.models import success_predictor
model = success_predictor.train_model(features, targets)

# Evaluate performance
from src.models.evaluation import metrics
metrics.evaluate_model(model, test_data)
```

### 4. Analysis & Insights
- Statistical correlation analysis
- 25-year trend identification
- Clustering of successful ad patterns
- Specific recommendations for Rogue Ridge

---

## Key Features

### Technology Stack
- **Languages**: Python, JavaScript
- **ML/AI**: scikit-learn, transformers, NLTK, spaCy
- **Data**: pandas, SQLite, Google Gemini API
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: Jupyter, Cursor IDE, GitHub

### Core Capabilities
- Multi-source data collection and integration
- Advanced sentiment analysis and NLP
- Predictive modeling for advertisement success
- Interactive dashboards and visualizations
- Automated report generation
- Business-ready recommendations

---

## Analysis Modules

### Statistical Analysis
```
```

### Trend Analysis
```
```

### Success Prediction
```
```

---

## Visualization & Reporting

### Interactive Dashboard
```bash
python scripts/deploy_dashboard.py
# Access dashboard at http://localhost:8050
```

### Generate Reports
```bash
python scripts/generate_reports.py --client-report
```

### Jupyter Analysis
```bash
jupyter notebook notebooks/05_analysis.ipynb
```

---

## Testing

Run the complete test suite:
```bash
# Run all tests
python -m pytest tests/

# Test specific modules
python -m pytest tests/test_data_collection/
python -m pytest tests/test_models/

# Generate coverage report
python -m pytest --cov=src tests/
>>>>>>> 72087d38e23fdea35548f560e950e62eab3d0d58
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

<<<<<<< HEAD
=======
---

## Configuration

### API Keys Setup
```yaml
# config/api_keys.yaml
youtube:
  api_key: "your_youtube_api_key"
  
reddit:
  client_id: "your_reddit_client_id"
  client_secret: "your_reddit_client_secret"
  
google_cloud:
  project_id: "your_project_id"
  gemini_api_key: "your_gemini_key"
```

### Database Configuration
```
```

---

## Usage Examples

### Data Collection
```python
from src.data_collection import youtube_scraper, reddit_scraper

# Collect YouTube data
youtube_data = youtube_scraper.collect_ad_data(video_ids)

# Collect Reddit discussions
reddit_data = reddit_scraper.search_discussions(keywords=["Super Bowl", "commercial"])
```

### Analysis
```
```

### Visualization
```
```
>>>>>>> 72087d38e23fdea35548f560e950e62eab3d0d58

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


<<<<<<< HEAD
=======
---

## Acknowledgments

- **Northlight Media** for the business case study
- **Google Cloud** for student credits and API access
- **Reddit** and **YouTube** for data access
- **Open Source Community** for the amazing libraries used

---

## Project Status

![Data Collection](https://img.shields.io/badge/Data%20Collection-In%20Progress-yellow)
![Model Training](https://img.shields.io/badge/Model%20Training-Pending-red)
![Documentation](https://img.shields.io/badge/Documentation-Up%20to%20Date-green)

**Last Updated**: [June 01 2025]  
**Current Phase**: Phase 1 - Project Planning  
**Next Milestone**: Data Collection Complete (June 15 2025)
>>>>>>> 72087d38e23fdea35548f560e950e62eab3d0d58
