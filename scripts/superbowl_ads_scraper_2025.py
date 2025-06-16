import os
import csv
import time
import logging
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SuperBowlAdsScraper:
    def __init__(self, year: int = 2025, chromedriver_path: str = None):
        """
        Initialize the Selenium WebDriver for Super Bowl Ads scraping
        
        Args:
            year (int): Year of Super Bowl ads to scrape
            chromedriver_path (str, optional): Path to ChromeDriver executable
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(f'superbowl_ads_scraper_{year}.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate and set ChromeDriver path
        self.chromedriver_path = self._find_chromedriver(chromedriver_path)
        
        # Setup Chrome options for more robust scraping
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        # Commented out headless mode for debugging
        # chrome_options.add_argument("--headless")  
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_argument('--disable-http2')
        chrome_options.add_argument('--disable-features=UseModernHttpProtocol')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Initialize WebDriver
        try:
            service = Service(self.chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise
        
        # Set year and base URLs
        self.year = year
        self.base_urls = [
            f'https://www.superbowl-ads.com/category/video/{year}_ads/',
            f'https://www.superbowl-ads.com/category/video/{year}_ads/page/2/'
        ]
        
        # Output directory setup
        self.output_dir = os.path.join(
            'data', 'raw', 'superbowl_ads', 
            f'{year}_YouTube_ID.csv'
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

    def _find_chromedriver(self, specified_path: str = None) -> str:
        """
        Find ChromeDriver executable path
        
        Args:
            specified_path (str, optional): User-specified path
        
        Returns:
            str: Path to ChromeDriver executable
        """
        # Possible ChromeDriver locations
        possible_paths = [
            specified_path,  # User-specified path (if provided)
            './chromedriver',  # Local directory
            './chromedriver.exe',
            os.path.join(os.path.dirname(__file__), 'chromedriver'),
            os.path.join(os.path.dirname(__file__), 'chromedriver.exe')
        ]
        
        # Find first existing path
        for path in possible_paths:
            if path and os.path.exists(path):
                self.logger.info(f"Using ChromeDriver from: {path}")
                return path
        
        # If no path found, raise an error with all checked paths
        raise FileNotFoundError(
            f"ChromeDriver not found. Checked paths:\n" + 
            "\n".join(str(path) for path in possible_paths if path) + 
            "\n\nPlease download from: https://sites.google.com/a/chromium.org/chromedriver/"
        )

    def scrape_ads(self) -> List[Dict[str, str]]:
        """
        Scrape Super Bowl ads for the specified year
        
        Returns:
            List of dictionaries containing ad details
        """
        all_ads = []
        
        try:
            for base_url in self.base_urls:
                self.logger.info(f"Scraping URL: {base_url}")
                self.driver.get(base_url)
                time.sleep(5)  # Increased wait time for page load
                
                # Find all ad articles
                ad_articles = self.driver.find_elements(
                    By.XPATH, 
                    '//div[contains(@class, "main-content-body")]//article'
                )
                
                self.logger.info(f"Found {len(ad_articles)} ads on this page")
                
                for index, article in enumerate(ad_articles, 1):
                    try:
                        # Find ad link within the article
                        ad_link = article.find_element(
                            By.XPATH, 
                            './/div[contains(@class, "entry-image")]/a'
                        )
                        ad_url = ad_link.get_attribute('href')
                        
                        # Navigate to ad detail page
                        self.driver.get(ad_url)
                        time.sleep(5)  # Increased wait time for page load
                        
                        # Extract YouTube video ID
                        youtube_meta = self.driver.find_elements(
                            By.XPATH, 
                            '//meta[@property="og:video"]'
                        )
                        
                        if youtube_meta:
                            youtube_url = youtube_meta[0].get_attribute('content')
                            youtube_id = youtube_url.split('/')[-1].split('?')[0]
                            
                            # Extract ad title
                            ad_title = self.driver.find_element(
                                By.XPATH, 
                                '//h1[contains(@class, "entry-title")]'
                            ).text
                            
                            ad_info = {
                                'year': self.year,
                                'title': ad_title,
                                'url': ad_url,
                                'youtube_id': youtube_id
                            }
                            
                            all_ads.append(ad_info)
                            self.logger.info(f"Scraped ad: {ad_title}")
                        
                        # Go back to previous page
                        self.driver.back()
                        time.sleep(2)
                    
                    except Exception as ad_error:
                        self.logger.error(f"Error scraping individual ad (index {index}): {ad_error}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            raise
        
        finally:
            self.driver.quit()
        
        return all_ads

    def save_to_csv(self, ads: List[Dict[str, str]]):
        """
        Save scraped ads to CSV file
        
        Args:
            ads (List[Dict[str, str]]): List of ad details
        """
        try:
            with open(self.output_dir, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['year', 'title', 'url', 'youtube_id']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for ad in ads:
                    writer.writerow(ad)
            
            self.logger.info(f"Saved {len(ads)} ads to {self.output_dir}")
        
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")

def main():
    """
    Main function to run the Super Bowl Ads scraper
    """
    try:
        # If ChromeDriver is in the same directory or in system PATH
        scraper = SuperBowlAdsScraper(year=2025)
        ads = scraper.scrape_ads()
        scraper.save_to_csv(ads)
        
        print(f"Successfully scraped {len(ads)} ads!")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()