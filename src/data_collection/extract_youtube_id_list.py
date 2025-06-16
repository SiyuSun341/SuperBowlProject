import pandas as pd 
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os

def find_chromedriver():
    """
    Find ChromeDriver executable path
    
    Returns:
        str: Path to ChromeDriver executable
    """
    # Possible ChromeDriver locations
    possible_paths = [
        'chromedriver',  # Default
        'chromedriver.exe',
        os.path.join(os.path.dirname(__file__), 'chromedriver'),
        os.path.join(os.path.dirname(__file__), 'chromedriver.exe'),
        '/usr/local/bin/chromedriver',  # Common Unix location
        '/usr/bin/chromedriver'
    ]
    
    # Find first existing path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no path found, suggest download
    raise FileNotFoundError(
        "ChromeDriver not found. Please download from: https://sites.google.com/a/chromium.org/chromedriver/ "
        "and place in the same directory or in system PATH."
    )

# ===== Path Configuration =====
CHROMEDRIVER_PATH = find_chromedriver()
CSV_PATH = os.path.join('data', 'raw', 'superbowl_ads', 'SuperBowl_Ads_Links.csv')
OUTPUT_PATH = CSV_PATH.replace(".csv", "_with_youtube_ids.csv")

# ===== Load CSV =====
df = pd.read_csv(CSV_PATH)
if 'youtube_id' not in df.columns:
    df['youtube_id'] = ''

# ===== Selenium Configuration Function =====
def create_driver():
    """
    Create and configure Chrome WebDriver
    
    Returns:
        WebDriver: Configured Chrome WebDriver
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument("--headless")  # Enable if page visualization is not needed
    service = Service(CHROMEDRIVER_PATH)
    return webdriver.Chrome(service=service, options=chrome_options)

# ===== Main Loop =====
for idx, row in df.iterrows():
    url = row['url']
    print(f"\n🔍 Processing link {idx+1}/{len(df)}: {url}")

    driver = create_driver()  # Open new browser

    try:
        driver.get(url)
        time.sleep(3)  # Wait for page load

        # Locate input and extract value
        hidden_input = driver.find_element(By.NAME, "main_video_url")
        youtube_id = hidden_input.get_attribute("value")
        df.at[idx, 'youtube_id'] = youtube_id
        print(f"  ✅ Extraction successful: {youtube_id}")

    except Exception as e:
        print(f"  ⚠ Extraction failed: {e}")

    finally:
        driver.quit()

    # ✅ Immediately save current state to file
    try:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"  💾 Progress saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"  ⚠ Save failed: {e}")

    # Random wait to prevent crawling detection
    sleep_time = random.randint(1, 5)
    print(f"  ⏳ Waiting {sleep_time} seconds...\n")
    time.sleep(sleep_time)

# ===== Final Completion =====
print(f"\n✅ All links processed, final results saved to: {OUTPUT_PATH}")