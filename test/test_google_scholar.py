#!/usr/bin/env python3
"""
Test script for Google Scholar abstract extraction with mirror support.
"""

from src.dblp_api import get_paper_list
from src.abstract_extractor import AbstractExtractor
from src.google_scholar_extractor import GoogleScholarExtractor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_scholar_mirrors():
    """Test different Google Scholar mirrors."""
    print("🌐 Testing Google Scholar Mirrors")
    print("=" * 40)
    
    mirrors = ['default', 'mirror1', 'mirror2', 'mirror3', 'semantic']
    
    for mirror in mirrors:
        print(f"\n🔍 Testing mirror: {mirror}")
        try:
            with GoogleScholarExtractor(headless=True, preferred_mirror=mirror) as scholar:
                # Test with a simple search
                abstract = scholar.search_paper(
                    "Attention Is All You Need", 
                    ["Ashish Vaswani", "Noam Shazeer"]
                )
                
                if abstract:
                    print(f"✅ {mirror}: SUCCESS ({len(abstract)} chars)")
                    print(f"Preview: {abstract[:100]}...")
                else:
                    print(f"❌ {mirror}: No abstract found")
        except Exception as e:
            print(f"💥 {mirror}: ERROR - {str(e)}")

def test_scholar_extraction():
    """Test Google Scholar extraction with DBLP papers."""
    print("\n📚 Testing Google Scholar with DBLP Papers")
    print("=" * 50)
    
    # Get papers from ASPLOS
    papers = get_paper_list("ASPLOS", 2023, filter_conference_papers=True)
    if not papers:
        print("❌ No papers found!")
        return
    
    test_papers = papers[:2]  # Test first 2 papers
    
    print(f"Testing {len(test_papers)} papers...")
    
    # Test with Google Scholar enabled
    extractor = AbstractExtractor(
        headless=False,  # Visible for debugging
        use_google_scholar=True,
        scholar_mirror='default'
    )
    
    try:
        results = extractor.extract_abstracts_batch(test_papers, delay=3.0)
        
        print(f"\n📊 Results:")
        for i, paper in enumerate(results, 1):
            print(f"\n{i}. {paper.title[:80]}...")
            if paper.abstract:
                print(f"   ✅ Abstract ({len(paper.abstract)} chars): {paper.abstract[:150]}...")
            else:
                print(f"   ❌ No abstract found")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"💥 Error: {e}")

def test_manual_search():
    """Test manual paper search."""
    print("\n🔍 Manual Paper Search Test")
    print("=" * 30)
    
    title = input("Enter paper title to search: ").strip()
    if not title:
        print("No title provided!")
        return
    
    authors_input = input("Enter authors (comma-separated, optional): ").strip()
    authors = [a.strip() for a in authors_input.split(',')] if authors_input else []
    
    mirror = input("Enter mirror (default/mirror1/mirror2/semantic): ").strip() or 'default'
    
    print(f"\n🔍 Searching for: {title}")
    print(f"Authors: {authors}")
    print(f"Mirror: {mirror}")
    
    try:
        with GoogleScholarExtractor(headless=False, preferred_mirror=mirror) as scholar:
            abstract = scholar.search_paper(title, authors)
            
            if abstract:
                print(f"\n✅ SUCCESS!")
                print(f"Abstract ({len(abstract)} chars):")
                print("-" * 40)
                print(abstract)
                print("-" * 40)
            else:
                print(f"\n❌ No abstract found")
    
    except Exception as e:
        print(f"💥 ERROR: {e}")

def main():
    print("🧪 Google Scholar Abstract Extraction Test Suite")
    print("=" * 50)
    
    while True:
        print("\nChoose test:")
        print("1. Test Google Scholar mirrors")
        print("2. Test with DBLP papers")
        print("3. Manual paper search")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            test_scholar_mirrors()
        elif choice == "2":
            test_scholar_extraction()
        elif choice == "3":
            test_manual_search()
        elif choice == "4":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()