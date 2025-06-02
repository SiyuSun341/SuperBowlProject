# Google News Scraper for Super Bowl Ads

This script collects news articles related to Super Bowl advertisements from Google News.

## Features

- Collects news articles about Super Bowl ads, commercials, and advertising
- Extracts article title, source, link, and publication time
- Saves data in JSON format
- Includes error handling and rate limiting
- Respects website's terms of service with appropriate delays

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python google_news_scraper.py
   ```

## Output

The script will create a `data` directory and save the collected articles in a JSON file with the following structure:

```json
[
  {
    "title": "Article Title",
    "source": "News Source",
    "link": "Article URL",
    "published_time": "Publication Time",
    "scraped_time": "Scraping Time"
  }
]
```

## Notes

- The script includes random delays between requests to avoid being blocked
- It uses a realistic User-Agent header
- Articles are collected from multiple search queries to ensure diverse coverage
- The script handles errors gracefully and continues collecting data even if some articles fail to process

## Disclaimer

This script is for educational purposes only. Please respect Google News' terms of service and robots.txt when using this scraper. 