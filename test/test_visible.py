#!/usr/bin/env python3
"""
Simple test script to demonstrate visible browser mode for debugging.
"""

from src.dblp_api import get_paper_list
from src.abstract_extractor import AbstractExtractor
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("🔍 Testing Abstract Extraction with Visible Browser")
    print("=" * 50)
    
    # Get one paper to test
    print("Getting papers from ASPLOS 2023...")
    papers = get_paper_list("ASPLOS", 2023, filter_conference_papers=True)
    
    if not papers:
        print("❌ No papers found!")
        return
    
    test_paper = papers[0]  # Just test the first paper
    
    print(f"\n📄 Testing Paper:")
    print(f"Title: {test_paper.title}")
    print(f"URL: {test_paper.url}")
    print(f"Type: {test_paper.type}")
    
    print(f"\n🖥️  Opening browser window (you will see Chrome open)...")
    print("🕐 This will take a moment as the page loads with human-like behavior...")
    
    # Create extractor with visible browser
    extractor = AbstractExtractor(headless=False, debug=True)
    
    try:
        print(f"\n🚀 Starting extraction...")
        abstract = extractor.extract_abstract(test_paper.url)
        
        if abstract:
            print(f"\n✅ SUCCESS!")
            print(f"Abstract length: {len(abstract)} characters")
            print(f"Preview: {abstract[:200]}...")
        else:
            print(f"\n❌ FAILED to extract abstract")
            print("Check the browser window and debug output above for details")
    
    except Exception as e:
        print(f"\n💥 ERROR: {str(e)}")
    
    finally:
        # Keep the browser open for a moment so you can see the final state
        input("\n⏸️  Press Enter to close the browser and exit...")
        extractor._teardown_driver()

if __name__ == "__main__":
    main()