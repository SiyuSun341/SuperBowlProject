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
superbowl-ad-analysis/
├── config/                       # Configuration files
├── data/                         # Data storage (raw, processed, samples)
├── src/                          # Source code modules
│   ├── data_collection/          # YouTube, Reddit, News scrapers
│   ├── data_processing/          # Text cleaning, sentiment analysis
│   ├── models/                   # ML models and evaluation
│   ├── analysis/                 # Statistical and trend analysis
│   ├── visualization/            # Charts and dashboards
│   └── utils/                    # Database, logging, helpers
├── notebooks/                    # Jupyter analysis notebooks
├── scripts/                      # Automation scripts
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── reports/                      # Deliverables and findings
└── models/                       # Trained model artifacts
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