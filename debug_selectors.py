#!/usr/bin/env python3
"""
Debug script to test selector issues on actual IEEE and ACM pages.
"""

from src.dblp_api import get_paper_list
from src.abstract_extractor import AbstractExtractor
import logging

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_single_paper(paper, extractor):
    """Test extraction on a single paper with detailed debugging."""
    print(f"\n{'='*60}")
    print(f"Testing: {paper.title}")
    print(f"URL: {paper.url}")
    print(f"DOI: {paper.doi}")
    print(f"Type: {paper.type}")
    print(f"{'='*60}")
    
    try:
        # Test the extraction
        abstract = extractor.extract_abstract(paper.url)
        
        if abstract:
            print(f"✅ SUCCESS: Extracted {len(abstract)} characters")
            print(f"Preview: {abstract[:200]}...")
        else:
            print("❌ FAILED: No abstract extracted")
            
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
    
    return abstract is not None

def main():
    print("🔍 Debug Script: Testing Abstract Extraction Selectors")
    print("This script will test actual papers and show detailed debugging info")
    
    # Get some conference papers
    venue = "ASPLOS"
    year = 2023
    
    print(f"\nGetting papers from {venue} {year}...")
    papers = get_paper_list(venue, year, filter_conference_papers=True)
    
    if not papers:
        print("❌ No papers found!")
        return
    
    print(f"Found {len(papers)} conference papers")
    
    # Test first few papers
    test_papers = papers[:3]  # Test first 3 papers
    
    print(f"\nTesting first {len(test_papers)} papers with DEBUG logging enabled...")
    
    # Initialize extractor with visible browser for debugging
    extractor = AbstractExtractor(headless=False, timeout=30)  # Visible browser, longer timeout
    
    success_count = 0
    
    try:
        for i, paper in enumerate(test_papers, 1):
            print(f"\n📄 Paper {i}/{len(test_papers)}")
            
            success = test_single_paper(paper, extractor)
            if success:
                success_count += 1
            
            # Wait between papers
            input("\nPress Enter to continue to next paper (or Ctrl+C to stop)...")
    
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    
    finally:
        extractor._teardown_driver()
        
        print(f"\n📊 Results: {success_count}/{len(test_papers)} papers successfully extracted")
        
        if success_count == 0:
            print("""
🚨 No abstracts were extracted! Possible issues:
1. Selectors are outdated/incorrect
2. Pages are being blocked (anti-bot detection)
3. Dynamic content loading issues
4. Page structure has changed

Check the debug output above for specific error details.
""")

if __name__ == "__main__":
    main()