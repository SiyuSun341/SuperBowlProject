"""
Super Bowl Ad Scraper Debugging Version
Used to check website content and extraction patterns
Save as: debug_scraper.py
"""

import sys
import os
from pathlib import Path
import time
import random
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def init_debug_driver():
    """Initialize Chrome driver for debugging"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    chrome_options.add_argument(f'--user-agent={user_agent}')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def debug_website_content():
    """Debug website content"""
    print("🔍 Debugging Super Bowl Ads Website")
    print("=" * 50)
    
    driver = init_debug_driver()
    
    try:
        # Test 2025 page
        url = "https://www.superbowl-ads.com/category/video/2025_ads/"
        print(f"🌐 Visiting: {url}")
        
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get page information
        page_title = driver.title
        page_source = driver.page_source
        current_url = driver.current_url
        
        print(f"📄 Page Title: {page_title}")
        print(f"🔗 Actual URL: {current_url}")
        print(f"📏 Page Size: {len(page_source)} characters")
        
        # Parse HTML
        soup = BeautifulSoup(page_source, 'html.parser')
        page_text = soup.get_text()
        
        print(f"📝 Page Text Length: {len(page_text)} characters")
        
        # Check keywords
        superbowl_count = page_text.lower().count('super bowl')
        ad_count = page_text.lower().count('ad')
        print(f"🏈 'Super Bowl' Occurrences: {superbowl_count}")
        print(f"📺 'Ad' Occurrences: {ad_count}")
        
        # Display page beginning content
        print("\n📋 Page Beginning Content (first 1000 characters):")
        print("-" * 40)
        print(page_text[:1000])
        print("-" * 40)
        
        # Search for ad-related lines
        print("\n🔍 Searching for Ad-related Content:")
        lines = page_text.split('\n')
        ad_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and ('super bowl' in line.lower() and 'ad' in line.lower()):
                if len(line) > 10:  # Ignore very short lines
                    ad_lines.append((i, line))
        
        print(f"Found {len(ad_lines)} lines with ad information:")
        for i, (line_num, line) in enumerate(ad_lines[:10]):  # Show first 10 lines
            print(f"  {i+1}. (Line {line_num}): {line[:100]}...")
        
        # Check HTML structure
        print("\n🏗️ HTML Structure Analysis:")
        
        # Find possible containers
        containers = soup.find_all(['div', 'article', 'section', 'li', 'h1', 'h2', 'h3'])
        ad_containers = []
        
        for container in containers:
            text = container.get_text().strip()
            if text and 'super bowl' in text.lower() and 'ad' in text.lower() and len(text) > 20:
                ad_containers.append({
                    'tag': container.name,
                    'class': container.get('class', []),
                    'id': container.get('id', ''),
                    'text': text[:150] + '...' if len(text) > 150 else text
                })
        
        print(f"Found {len(ad_containers)} possible ad containers:")
        for i, container in enumerate(ad_containers[:5]):
            print(f"  {i+1}. <{container['tag']} class='{container['class']}'")
            print(f"      {container['text']}")
        
        # Find YouTube links
        print("\n🔗 YouTube Link Analysis:")
        youtube_links = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            if 'youtube.com' in href or 'youtu.be' in href:
                youtube_links.append(href)
        
        print(f"Found {len(youtube_links)} YouTube links:")
        for i, link in enumerate(youtube_links[:5]):
            print(f"  {i+1}. {link}")
        
        # Save debug information
        debug_file = "debug_page_content.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(page_source)
        print(f"\n💾 Page content saved to: {debug_file}")
        
        # Try different extraction patterns
        print("\n🧪 Testing Different Extraction Patterns:")
        
        patterns = [
            r'([A-Za-z\s&\.\'%-]+?)\s+Super\s+Bowl\s+.*?2025.*?Ad.*?"([^"]+)"',
            r'([A-Za-z\s&\.\'%-]+?)\s+Super\s+Bowl\s+.*?Ad.*?"([^"]+)"',
            r'([A-Za-z\s&\.\'%-]+?)\s+2025\s+Super\s+Bowl.*?"([^"]+)"',
            r'Super\s+Bowl\s+.*?2025.*?Ad.*?"([^"]+)".*?([A-Za-z\s&\.\'%-]+)',
            r'([A-Za-z]+)[^"]*Super\s+Bowl[^"]*Ad[^"]*"([^"]+)"'
        ]
        
        for i, pattern in enumerate(patterns, 1):
            try:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                print(f"  Pattern {i}: Found {len(matches)} matches")
                for j, match in enumerate(matches[:3]):
                    if isinstance(match, tuple) and len(match) >= 2:
                        print(f"    {j+1}. Brand: '{match[0][:30]}' Title: '{match[1][:30]}'")
            except Exception as e:
                print(f"  Pattern {i}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debugging Failed: {e}")
        return False
        
    finally:
        driver.quit()

def test_simple_extraction():
    """Test simple text extraction"""
    print("\n🧪 Testing Simplified Extraction Method")
    print("=" * 30)
    
    driver = init_debug_driver()
    
    try:
        url = "https://www.superbowl-ads.com/category/video/2025_ads/"
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find specific elements
        elements_to_try = [
            'h1', 'h2', 'h3', 'h4',
            '.entry-title', '.post-title', '.ad-title',
            '[class*="title"]', '[class*="ad"]', '[class*="video"]'
        ]
        
        for selector in elements_to_try:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"✓ Found {len(elements)} '{selector}' elements:")
                    for i, elem in enumerate(elements[:3]):
                        text = elem.text.strip()
                        if text:
                            print(f"  {i+1}. {text[:80]}...")
            except:
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified Extraction Test Failed: {e}")
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    print("🏈 Super Bowl Ad Scraper Debugging Tool")
    print("🔧 Used to Diagnose Data Extraction Issues")
    print("=" * 60)
    
    # Run debugging
    success1 = debug_website_content()
    success2 = test_simple_extraction()
    
    if success1 or success2:
        print("\n✅ Debugging Completed!")
        print("📋 Please check the output above to understand website structure")
        print("💡 Based on debugging results, we can optimize extraction patterns")
    else:
        print("\n❌ Debugging Failed")
        print("🔧 Possible network issues or website access restrictions")