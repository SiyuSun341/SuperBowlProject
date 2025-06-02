import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import time
import random

class GoogleNewsScraper:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://news.google.com/search"
        
    def search_news(self, query, num_articles=10):
        """
        Search Google News for articles related to the query
        """
        articles = []
        page = 1
        
        while len(articles) < num_articles:
            # Construct the search URL
            params = {
                'q': query,
                'hl': 'en-US',
                'gl': 'US',
                'ceid': 'US:en'
            }
            
            try:
                # Add random delay to avoid being blocked
                time.sleep(random.uniform(2, 4))
                
                # Make the request
                response = requests.get(self.base_url, params=params, headers=self.headers)
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all article elements
                article_elements = soup.find_all('article')
                
                for article in article_elements:
                    if len(articles) >= num_articles:
                        break
                        
                    try:
                        # Extract article information
                        title_element = article.find('h3')
                        if not title_element:
                            continue
                            
                        title = title_element.text.strip()
                        
                        # Get the article link
                        link_element = article.find('a')
                        if not link_element:
                            continue
                            
                        link = link_element.get('href')
                        if link.startswith('./'):
                            link = 'https://news.google.com' + link[1:]
                            
                        # Get the source and time
                        source_element = article.find('div', {'class': 'UPmit'})
                        source = source_element.text.strip() if source_element else "Unknown Source"
                        
                        time_element = article.find('time')
                        time_str = time_element.get('datetime') if time_element else None
                        
                        # Create article data
                        article_data = {
                            'title': title,
                            'source': source,
                            'link': link,
                            'published_time': time_str,
                            'scraped_time': datetime.now().isoformat()
                        }
                        
                        articles.append(article_data)
                        
                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        continue
                
                page += 1
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break
                
        return articles
    
    def save_articles(self, articles, filename):
        """
        Save articles to a JSON file
        """
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(articles)} articles to {filepath}")

def main():
    # Create save directory
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Initialize scraper
    scraper = GoogleNewsScraper(save_dir)
    
    # Search queries
    queries = [
        "Super Bowl Ads",
        "Super Bowl Commercials",
        "Super Bowl Advertising"
    ]
    
    # Collect articles for each query
    all_articles = []
    for query in queries:
        print(f"\nSearching for: {query}")
        articles = scraper.search_news(query, num_articles=10)
        all_articles.extend(articles)
        
    # Save all articles
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scraper.save_articles(all_articles, f'superbowl_news_{timestamp}.json')

if __name__ == "__main__":
    main() 