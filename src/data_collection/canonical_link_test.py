from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os

# Find ChromeDriver path
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

# Find ChromeDriver path
CHROMEDRIVER_PATH = find_chromedriver()

# Set browser options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")  # Optional: Maximize window
# options.add_argument("--headless")  # Optional: Headless mode

# Start browser service
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Target URL
url = "https://www.superbowl-ads.com/2000-wwf-beauty-pagent/"
driver.get(url)

# Wait for page load
time.sleep(3)

# Locate input element and extract value
try:
    hidden_input = driver.find_element(By.NAME, "main_video_url")
    value = hidden_input.get_attribute("value")
    print(f"✅ Extraction successful, YouTube ID is: {value}")
except Exception as e:
    print("❌ Specified <input> element not found:", e)

# Close browser
driver.quit()