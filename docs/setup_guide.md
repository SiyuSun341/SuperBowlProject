# Setup Guide - Super Bowl Advertisement Analysis

This guide provides detailed instructions for setting up the complete environment for the Super Bowl Advertisement Analysis project.

## Table of Contents
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Dependencies Installation](#dependencies-installation)
- [API Configuration](#api-configuration)
- [External Tools Setup](#external-tools-setup)
- [Data Directory Structure](#data-directory-structure)
- [GPU Configuration](#gpu-configuration)
- [Testing Your Setup](#testing-your-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 11+
- **CPU**: Intel i5/AMD Ryzen 5 or better (8+ cores recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 100GB free space (500GB recommended for full dataset)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but highly recommended)
- **Python**: 3.8 - 3.10 (3.11+ may have compatibility issues)

### Recommended Specifications
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9 (16+ cores)
- **RAM**: 32GB or more
- **Storage**: 1TB SSD
- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM)

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SiyuSun341/SuperBowlProject.git
cd SuperBowlProject
```

### 2. Create Python Virtual Environment

#### Windows
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation (should show Python from .venv)
where python
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation
which python
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

## Dependencies Installation

### 1. Core Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
# Data Collection & Web Scraping
pip install selenium==4.15.0
pip install beautifulsoup4==4.12.2
pip install requests==2.31.0
pip install praw==7.7.1
pip install yt-dlp==2023.11.16
pip install fake-useragent==1.4.0

# Data Processing & Analysis
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.1
pip install seaborn==0.13.0

# NLP & Sentiment Analysis
pip install textblob==0.17.1
pip install vaderSentiment==3.3.2
pip install transformers==4.35.2
pip install torch==2.1.1
pip install youtube-transcript-api==0.6.1

# Machine Learning
pip install catboost==1.2.2
pip install statsmodels==0.14.0

# Audio/Video Processing
pip install whisper==1.1.10
pip install librosa==0.10.1
pip install opencv-python==4.8.1.78

# Google Cloud & APIs
pip install google-generativeai==0.3.0
pip install google-cloud-storage==2.10.0
pip install python-dotenv==1.0.0

# Utilities
pip install tqdm==4.66.1
pip install urllib3==2.1.0
```

### 2. Platform-Specific Dependencies

#### For GPU Support (NVIDIA)
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (adjust for your CUDA version)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## API Configuration

### 1. Create Configuration Directory
```bash
mkdir -p config
```

### 2. Create .env File
Create `config/.env` with the following template:

```env
# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_data_api_key_here

# Reddit API (PRAW)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=SuperBowlAdAnalysis/1.0 by YourUsername
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password

# Google Cloud / Gemini API
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Optional: OpenAI (if using GPT models)
OPENAI_API_KEY=your_openai_api_key
```

### 3. Obtain API Keys

#### YouTube Data API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Restrict key to YouTube Data API v3

#### Reddit API
1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill in:
   - Name: SuperBowlAdAnalysis
   - App type: Script
   - Redirect URI: http://localhost:8080
4. Note the client ID and secret

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. For service account (optional):
   - Go to Google Cloud Console
   - Create service account
   - Download JSON key file

## External Tools Setup

### 1. ChromeDriver (for Selenium)
```bash
# Check Chrome version
google-chrome --version  # Linux/Mac
# or open Chrome → Help → About Google Chrome

# Download matching ChromeDriver from https://chromedriver.chromium.org/
# Place in project root or system PATH

# Verify installation
chromedriver --version
```

### 2. FFmpeg (for video processing)

#### Windows
```cmd
# Download from https://ffmpeg.org/download.html
# Extract and add to PATH
# Or use Chocolatey:
choco install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Verify Installation
```bash
ffmpeg -version
```

### 3. Additional Tools
```bash
# Image processing
pip install Pillow==10.1.0

# Progress bars for long operations
pip install alive-progress==3.1.4
```

## Data Directory Structure

Create the following directory structure:

```
SuperBowlProject/
├── config/
│   ├── .env
│   └── prompts/
│       └── gemini_prompts.json
├── data/
│   ├── raw/
│   │   ├── superbowl_ads/
│   │   │   ├── SuperBowl_Ads_Links.csv
│   │   │   └── Youtube_ID_Yearly/
│   │   ├── ad_features/
│   │   │   ├── details/
│   │   │   └── comments/
│   │   └── videos/
│   ├── processed/
│   │   ├── features/
│   │   ├── sentiment/
│   │   └── multimodal/
│   └── results/
├── models/
│   ├── trained/
│   └── checkpoints/
├── logs/
├── scripts/
├── src/
└── deliverables/
```

Create directories:
```bash
mkdir -p data/{raw,processed,results}
mkdir -p data/raw/{superbowl_ads,ad_features,videos}
mkdir -p data/raw/ad_features/{details,comments}
mkdir -p data/processed/{features,sentiment,multimodal}
mkdir -p models/{trained,checkpoints}
mkdir -p logs deliverables
```

## GPU Configuration

### 1. Verify CUDA Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
```

### 2. Configure Whisper for GPU
```python
# In your scripts, use:
import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)
```

### 3. Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvidia-ml-py
pip install nvidia-ml-py3
```

## Testing Your Setup

### 1. Test Script (`test_setup.py`)
Create and run this test script:

```python
#!/usr/bin/env python3
"""Test script to verify environment setup"""

import sys
import subprocess

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def test_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {command}: {e}")
        return False

def main():
    print("Testing Python modules...")
    modules = [
        'selenium', 'beautifulsoup4', 'praw', 'yt_dlp',
        'pandas', 'numpy', 'sklearn', 'torch',
        'textblob', 'vaderSentiment', 'transformers',
        'whisper', 'catboost', 'google.generativeai'
    ]
    
    module_results = [test_import(mod) for mod in modules]
    
    print("\nTesting external tools...")
    commands = [
        'ffmpeg -version',
        'chromedriver --version'
    ]
    
    command_results = [test_command(cmd) for cmd in commands]
    
    print("\nTesting GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU detected (CPU mode will be used)")
    except:
        print("❌ GPU test failed")
    
    total_tests = len(modules) + len(commands)
    passed_tests = sum(module_results) + sum(command_results)
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests < total_tests:
        print("\n⚠️  Some components are missing. Please install missing dependencies.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Environment is ready.")

if __name__ == "__main__":
    main()
```

Run the test:
```bash
python test_setup.py
```

### 2. Test Data Collection
```bash
# Test YouTube ID extraction (small sample)
python scripts/extract_youtube_id_list.py --test --limit 5

# Test Reddit connection
python -c "import praw; reddit = praw.Reddit(client_id='test'); print('PRAW imported successfully')"
```

### 3. Test Processing Pipeline
```bash
# Test video processing with a sample video
python scripts/ffmpeg_process_videos.py --test-video sample.mp4
```

## Troubleshooting

### Common Issues

#### 1. ChromeDriver Version Mismatch
```
selenium.common.exceptions.SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version XX
```
**Solution**: Download the correct ChromeDriver version matching your Chrome browser.

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size in processing scripts
- Use smaller models (e.g., Whisper "small" instead of "medium")
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 3. API Rate Limits
```
prawcore.exceptions.TooManyRequests: received 429 HTTP response
```
**Solution**: 
- Implement exponential backoff
- Add delays between requests
- Use multiple API keys with rotation

#### 4. FFmpeg Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```
**Solution**: Ensure FFmpeg is installed and in system PATH.

#### 5. Import Errors
```
ModuleNotFoundError: No module named 'package_name'
```
**Solution**: 
- Ensure virtual environment is activated
- Reinstall the specific package: `pip install package_name`

### Getting Help

1. Check the [GitHub Issues](https://github.com/SiyuSun341/SuperBowlProject/issues)
2. Review error logs in the `logs/` directory
3. Contact: sunsiyu.suzy@gmail.com

## Next Steps

After successful setup:
1. Review the [User Guide](user_guide.md) for detailed usage instructions
2. Start with data collection: `python scripts/extract_youtube_id_list.py`
3. Follow the pipeline stages in order
4. Monitor progress in `logs/` directory

---

**Note**: This setup guide is for academic research purposes. Ensure compliance with all API terms of service and rate limits.
