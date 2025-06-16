import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import random
import re
import json
import pandas as pd
import logging
import numpy as np
import hashlib
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('super_bowl_ads_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SuperBowlAdsScraper:
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize Super Bowl Ad Scraper
        
        :param project_root: Project root directory path for data storage
        """
        # Set project root directory
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        
        # Create data storage directories
        self.data_dir = self.project_root / 'data' / 'raw' / 'superbowl_ads'
        self.reports_dir = self.project_root / 'reports' / 'technical_reports'
        self.logs_dir = self.project_root / 'logs'
        
        # Create directories
        for dir_path in [self.data_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Browser driver
        self.driver: Optional[webdriver.Chrome] = None
        
        # Ad data storage
        self.ads_data: List[Dict] = []
        
        # Year range
        self.start_year = 2000
        self.end_year = 2025
        
        # Anti-crawling strategy parameters
        self.RESTART_EVERY = 5
        self.WAIT_EVERY = 10
        self.MAX_RETRIES = 3
        self.request_count = 0
        self.restart_count = 0
        
        # Statistical variables
        self.total_ads_count = 0
        self.stats_by_year = {}
        
        # Random User-Agent pool
        self.USER_AGENTS = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    def generate_random_cookie(self) -> str:
        """Generate random cookie"""
        random_token = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
        return f"csrftoken={random_token}; sessionid=fake_{random.randint(100000, 999999)}"
    
    def get_pareto_delay(self, min_delay: float = 3.0, max_delay: float = 15.0) -> float:
        """Pareto distribution delay"""
        shape = 1.16
        pareto_sample = np.random.pareto(shape)
        delay = min_delay + pareto_sample * 3
        return min(delay, max_delay)

    def init_driver(self) -> webdriver.Chrome:
        """
        Initialize Selenium Chrome WebDriver
        
        :return: Chrome WebDriver instance
        """
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-images')
            chrome_options.add_argument('--disable-javascript')
            
            # Random User-Agent
            user_agent = random.choice(self.USER_AGENTS)
            chrome_options.add_argument(f'--user-agent={user_agent}')
            
            # Disable automation features
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Use webdriver-manager to manage ChromeDriver
            service = Service(ChromeDriverManager().install())
            
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Remove WebDriver features
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info(f"✅ Browser driver initialized successfully (attempt {self.restart_count + 1})")
            return driver
        except Exception as e:
            logger.error(f"❌ Browser driver initialization failed: {e}")
            raise

    def extract_youtube_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL
        
        :param url: YouTube video URL
        :return: YouTube video ID
        """
        if not url:
            return None
        
        youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(youtube_regex, url)
        return match.group(1) if match else None
        def scrape_year_ads(self, year: int) -> List[Dict]:
        """
        Scrape Super Bowl ads for a specific year
        
        :param year: Year to scrape
        :return: List of ads for the year
        """
        # Browser restart strategy
        if self.request_count % self.RESTART_EVERY == 0 or self.driver is None:
            if self.driver:
                self.driver.quit()
                time.sleep(random.uniform(3, 6))
            self.driver = self.init_driver()
            self.restart_count += 1
        
        # Build year page URL
        year_ads_url = f"https://www.superbowl-ads.com/category/video/{year}_ads/"
        
        year_ads = []
        page = 1
        
        try:
            while True:
                # Pagination handling
                if page > 1:
                    year_ads_url = f"https://www.superbowl-ads.com/category/video/{year}_ads/page/{page}/"
                
                self.driver.get(year_ads_url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cactus-post-item"))
                )
                
                # Random delay
                time.sleep(self.get_pareto_delay())
                
                # Find all ad articles
                ad_articles = self.driver.find_elements(By.CLASS_NAME, "cactus-post-item")
                
                # Exit loop if no ads found
                if not ad_articles:
                    break
                
                for article in ad_articles:
                    try:
                        # Extract details page link
                        ad_link = article.find_element(By.CSS_SELECTOR, "h3.cactus-post-title a")
                        ad_url = ad_link.get_attribute('href')
                        ad_title = ad_link.text.strip()
                        
                        # Enter details page to extract more information
                        self.driver.get(ad_url)
                        WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        
                        # Random delay
                        time.sleep(random.uniform(1, 3))
                        
                        # Use BeautifulSoup for more reliable page parsing
                        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                        
                        # Try to extract YouTube link
                        youtube_links = soup.select('a[href*="youtube.com"], iframe[src*="youtube.com"]')
                        youtube_url = None
                        
                        for link in youtube_links:
                            href = link.get('href') or link.get('src')
                            if href and 'youtube.com' in href:
                                youtube_url = href
                                break
                        
                        # Parse brand
                        brand_match = re.search(r'^(.*?)\s*\d{4}\s*Super\s*Bowl', ad_title, re.IGNORECASE)
                        brand = brand_match.group(1).strip() if brand_match else 'Unknown'
                        
                        ad_info = {
                            'year': year,
                            'brand': brand,
                            'title': ad_title,
                            'ad_url': ad_url,
                            'youtube_url': youtube_url,
                            'youtube_id': self.extract_youtube_id(youtube_url) if youtube_url else None,
                            'scraped_timestamp': datetime.now().isoformat()
                        }
                        
                        year_ads.append(ad_info)
                        
                        # Update statistics
                        self.total_ads_count += 1
                        if year not in self.stats_by_year:
                            self.stats_by_year[year] = {'total': 0, 'with_youtube': 0}
                        
                        self.stats_by_year[year]['total'] += 1
                        if youtube_url:
                            self.stats_by_year[year]['with_youtube'] += 1
                        
                        # Request count and random delay
                        self.request_count += 1
                        if self.request_count % self.WAIT_EVERY == 0:
                            cooldown = random.uniform(15, 30)
                            logger.info(f"🧊 Cooling down: {cooldown:.1f} seconds")
                            time.sleep(cooldown)
                    
                    except Exception as e:
                        logger.warning(f"Error processing ad {ad_url}: {e}")
                
                # Check for next page
                try:
                    next_buttons = self.driver.find_elements(By.CSS_SELECTOR, '.page-navigation .nav-next a')
                    if not next_buttons:
                        break
                    page += 1
                except:
                    break
        
        except Exception as e:
            logger.error(f"Failed to scrape ads for {year}: {e}")
        
        return year_ads
        def scrape_all_years(self, start_year: int = 2000, end_year: int = 2025, reverse_order: bool = True) -> pd.DataFrame:
        """
        Scrape Super Bowl ads for specified years
        
        :param start_year: Starting year
        :param end_year: Ending year
        :param reverse_order: Whether to scrape in reverse chronological order
        :return: DataFrame containing all ads
        """
        all_ads = []
        
        try:
            # Determine year list
            if reverse_order:
                year_list = list(range(end_year, start_year - 1, -1))
            else:
                year_list = list(range(start_year, end_year + 1))
            
            for year in year_list:
                logger.info(f"🏈 Starting to scrape Super Bowl ads for {year}...")
                year_ads = self.scrape_year_ads(year)
                all_ads.extend(year_ads)
                logger.info(f"✅ Scraped {len(year_ads)} ads for {year}")
                
                # Random delay to avoid blocking
                time.sleep(self.get_pareto_delay())
        
        except Exception as e:
            logger.error(f"Error during scraping process: {e}")
        
        finally:
            if self.driver:
                self.driver.quit()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ads)
        
        # Save as CSV and JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.data_dir / f'superbowl_ads_{timestamp}.csv'
        json_path = self.data_dir / f'superbowl_ads_{timestamp}.json'
        
        # Save data
        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        # Generate report
        self.generate_final_report(csv_path, json_path)
        
        logger.info(f"💾 Data saved:")
        logger.info(f"CSV: {csv_path}")
        logger.info(f"JSON: {json_path}")
        
        return df

    def generate_final_report(self, csv_path: Path, json_path: Path):
        """
        Generate final scraping report
        
        :param csv_path: CSV file path
        :param json_path: JSON file path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"superbowl_ads_scraping_report_{timestamp}.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("🏈 Super Bowl Ads Scraping Report\n")
                f.write("=" * 40 + "\n\n")
                
                # Data overview
                f.write("📊 Data Overview\n")
                f.write(f"├─ Total Ads: {self.total_ads_count}\n")
                f.write(f"├─ Years Processed: {len(self.stats_by_year)}\n")
                f.write(f"├─ Browser Restarts: {self.restart_count}\n")
                f.write(f"└─ Total Requests: {self.request_count}\n\n")
                
                # Year-wise statistics
                f.write("🗓️ Year-wise Statistics\n")
                for year in sorted(self.stats_by_year.keys()):
                    stats = self.stats_by_year[year]
                    youtube_rate = (stats['with_youtube'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    f.write(f"├─ {year}: {stats['total']} ads (YouTube: {stats['with_youtube']}, {youtube_rate:.1f}%)\n")
                
                # Output files
                f.write("\n💾 Output Files\n")
                f.write(f"├─ CSV: {csv_path}\n")
                f.write(f"└─ JSON: {json_path}\n\n")
                
                # Technical information
                f.write("🔧 Technical Information\n")
                f.write("├─ Data Collection Method: Selenium WebDriver\n")
                f.write("├─ Anti-Crawling Strategy: Random User-Agent, Dynamic Delay\n")
                f.write("├─ Browser Restart Mechanism: Restart every 5 requests\n")
                f.write("└─ Data Writing: Real-time CSV, Batch JSON\n")
            
            logger.info(f"📋 Report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {e}")

def main():
    scraper = SuperBowlAdsScraper()
    df = scraper.scrape_all_years()
    print(f"Total {len(df)} Super Bowl ads scraped")
    print(f"Data saved in {scraper.data_dir}")

if __name__ == "__main__":
    main()