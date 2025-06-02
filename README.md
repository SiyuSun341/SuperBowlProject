# Super Bowl Advertisement Analysis

## Project Overview

This project analyzes 25 years of Super Bowl commercials to identify key success factors for **Forge & Field's** new **Rogue Ridge** product line advertisement. As business analysts for Northlight Media, we leverage AI/ML technologies to analyze multi-source data and provide actionable insights for an $11 million advertising investment.

### Business Context
- **Client**: Forge & Field Brands (menswear, accessories, outwear)
- **Product**: Rogue Ridge personal care line (targeting American men)
- **Investment**: $11M total budget ($7-8M for Super Bowl airtime alone)
- **Goal**: Identify critical success factors to make or break the product line

---


## Quick Start

### Prerequisites
- Python 3.8+
- Git
- Google Cloud account (for Gemini API)
- Reddit API access
- YouTube Data API v3 key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/superbowl-ad-analysis.git
   cd superbowl-ad-analysis
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   cp config/api_keys.yaml.template config/api_keys.yaml
   # Edit api_keys.yaml with your actual API keys
   ```

5. **Initialize database**
   ```bash
   python scripts/setup_environment.py
   ```

### First Run
```bash
# Test data collection
python scripts/run_data_collection.py --test

# Run full analysis pipeline
python scripts/train_models.py
python scripts/generate_reports.py
```

---

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
```python
from src.analysis import statistical_analysis
stats = statistical_analysis.analyze_correlations(data)
```

### Trend Analysis
```python
from src.analysis import trend_analysis
trends = trend_analysis.identify_patterns(yearly_data)
```

### Success Prediction
```python
from src.models import success_predictor
predictions = success_predictor.predict_success(ad_features)
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
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Documentation](docs/api_documentation.md) | Complete API reference |
| [Data Dictionary](docs/data_dictionary.md) | Field definitions and schemas |
| [Model Documentation](docs/model_documentation.md) | ML model details and performance |
| [Setup Guide](docs/setup_guide.md) | Detailed installation instructions |
| [User Guide](docs/user_guide.md) | Usage examples and workflows |

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
```yaml
# config/database_config.yaml
sqlite:
  database_path: "database/superbowl_ads.db"
  backup_interval: 3600  # seconds
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
```python
from src.analysis import insight_generator

# Generate business insights
insights = insight_generator.analyze_success_factors(combined_data)
print(insights.get_recommendations_for_rogue_ridge())
```

### Visualization
```python
from src.visualization import dashboard

# Create interactive charts
dashboard.create_trend_analysis_chart(data)
dashboard.create_success_factor_heatmap(correlations)
```

---

## Success Metrics

### Data Quality Targets
- Data completeness: >95%
- Duplicate rate: <5%
- 500+ advertisements analyzed
- 500K+ comments processed

### Model Performance Goals
- Sentiment analysis accuracy: >85%
- Success prediction AUC: >0.8
- Feature importance clearly interpretable
- Cross-validation performance consistent

### Business Impact
- 5-7 key success factors identified
- Actionable recommendations for Rogue Ridge
- Quantifiable ROI projections
- Competitive positioning strategy

---

## Contributing

### Development Workflow
1. Create feature branch: `git checkout -b feature/your-feature`
2. Write tests for new functionality
3. Ensure all tests pass: `python -m pytest`
4. Update documentation if needed
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Maintain test coverage >80%
- Use type hints where applicable

---

## Support & Contact


### Technical Issues
- Report bugs via GitHub Issues
- Email: [sunsiyu.suzy@gmail.com]

### Resources
- [Materials](link-to-course)
- [Demo Videos](link-to-demos)
- [Discussion Forum](link-to-forum)

---

## License

This project is created for academic purposes as part of MBT-capstone at Purdue University. 

**Note**: This is a private repository for educational use. Do not distribute without permission.

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
