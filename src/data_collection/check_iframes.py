from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
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

# ✅ Set ChromeDriver path
CHROMEDRIVER_PATH = find_chromedriver()

# ✅ Set launch parameters (window visible)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")  # Maximize browser window
# options.add_argument("--headless")  # Enable headless mode if you don't want to show interface

# ✅ Start browser service and create driver instance
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# ✅ Open webpage (can be replaced with any Super Bowl ad page)
url = "https://www.superbowl-ads.com/fanduel-super-bowl-lix-2025-ad-the-kick-of-destiny-3-recap-of-eli-vs-peyton-manning/"
driver.get(url)

# ✅ Wait 5 seconds to ensure JS lazy loading completes
time.sleep(5)

# ✅ Find all elements on the page
all_elements = driver.find_elements(By.xpath, "//*")

print(f"\n[✔] Total {len(all_elements)} elements found on the page:\n")

# ✅ Iterate through all elements and output tag names and key attributes
for idx, element in enumerate(all_elements[:50]):  # Limit output to first 50 elements to avoid excessive output
    tag_name = element.tag_name
    id_attr = element.get_attribute("id")
    class_attr = element.get_attribute("class")
    print(f"[{idx+1}] Tag: <{tag_name}>, id: '{id_attr}', class: '{class_attr}'")

# ✅ Close browser
driver.quit()