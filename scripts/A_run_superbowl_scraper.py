#!/usr/bin/env python3
"""
Super Bowl Ad Scraper Execution Script
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.A_superbowl_ads_scraper import SuperBowlAdsScraper

def main():
    parser = argparse.ArgumentParser(description="Super Bowl Ad Scraper")
    parser.add_argument("--start-year", type=int, default=2000, help="Start year")
    parser.add_argument("--end-year", type=int, default=2025, help="End year")
    parser.add_argument("--test", action="store_true", help="Test mode (only scrape last 3 years)")
    parser.add_argument("--order", choices=['new-to-old', 'old-to-new'], 
                       default='new-to-old', help="Scraping order")
    
    args = parser.parse_args()
    
    print("🏈 Super Bowl Ad Scraper")
    print("=" * 50)
    
    try:
        # Initialize scraper
        scraper = SuperBowlAdsScraper(project_root=project_root)
        
        # Set parameters
        if args.test:
            start_year = max(2000, args.end_year - 2)
            end_year = args.end_year
            print(f"🧪 Test Mode: Scraping data from {start_year}-{end_year}")
        else:
            start_year = args.start_year
            end_year = args.end_year
        
        # Determine scraping order
        reverse_order = (args.order == 'new-to-old')
        
        # Start scraping
        print(f"\n🚀 Starting scraping data from {start_year}-{end_year}...")
        df = scraper.scrape_all_years(start_year, end_year, reverse_order)
        
        if len(df) > 0:
            print(f"\n🎉 Scraping completed!")
            print(f"📊 Total: {len(df)} ads")
            
            # Display sample
            print("\n📋 Data Sample:")
            sample_df = df[['year', 'brand', 'title']].head(10)
            print(sample_df.to_string(index=False))
        else:
            print("❌ No data collected")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
        return 1
    except Exception as e:
        print(f"\n💥 Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())