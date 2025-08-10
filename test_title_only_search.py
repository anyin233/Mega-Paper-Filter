#!/usr/bin/env python3
"""
Test the simplified Google Scholar behavior: title-only search with results extraction.
"""

from src.google_scholar_extractor import GoogleScholarExtractor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_title_only_search():
    """Test the new title-only search behavior."""
    print("🔍 Testing Simplified Google Scholar Search")
    print("=" * 45)
    print("New behavior: Search with title only, extract from results page")
    print()
    
    # Test papers with known abstracts
    test_papers = [
        {
            'title': 'Attention Is All You Need',
            'expected': 'transformer'  # Should contain this word
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'expected': 'bidirectional'
        },
        {
            'title': 'Deep Residual Learning for Image Recognition',
            'expected': 'residual'
        }
    ]
    
    mirrors_to_test = ['default', 'mirror1', 'semantic']
    
    for mirror in mirrors_to_test:
        print(f"\n🌐 Testing mirror: {mirror}")
        print("-" * 30)
        
        try:
            with GoogleScholarExtractor(headless=False, preferred_mirror=mirror) as scholar:
                for i, paper in enumerate(test_papers, 1):
                    print(f"\n📄 Paper {i}: {paper['title'][:50]}...")
                    
                    abstract = scholar.search_paper(paper['title'])
                    
                    if abstract:
                        print(f"✅ SUCCESS ({len(abstract)} chars)")
                        
                        # Check if expected content is present
                        if paper['expected'].lower() in abstract.lower():
                            print(f"✓ Content validation passed (found '{paper['expected']}')")
                        else:
                            print(f"⚠️ Content validation failed ('{paper['expected']}' not found)")
                        
                        print(f"Preview: {abstract[:150]}...")
                    else:
                        print(f"❌ No abstract found")
                
                # Wait for user to see results
                input(f"\nPress Enter to continue to next mirror...")
        
        except Exception as e:
            print(f"💥 Error with {mirror}: {e}")
            continue

def test_dblp_papers():
    """Test with actual DBLP papers."""
    print("\n📚 Testing with DBLP Papers")
    print("=" * 30)
    
    try:
        from src.dblp_api import get_paper_list
        
        papers = get_paper_list("ASPLOS", 2023, filter_conference_papers=True)
        if not papers:
            print("❌ No DBLP papers found!")
            return
        
        test_papers = papers[:2]  # Test first 2 papers
        
        print(f"Testing {len(test_papers)} papers from ASPLOS 2023:")
        
        with GoogleScholarExtractor(headless=False, preferred_mirror='default') as scholar:
            for i, paper in enumerate(test_papers, 1):
                print(f"\n📄 Paper {i}: {paper.title[:60]}...")
                print(f"Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                
                abstract = scholar.search_paper(paper.title)  # Title-only search
                
                if abstract:
                    print(f"✅ SUCCESS ({len(abstract)} chars)")
                    print(f"Abstract: {abstract[:200]}...")
                else:
                    print(f"❌ No abstract found")
                
                input("Press Enter for next paper...")
    
    except ImportError:
        print("❌ Could not import DBLP API")
    except Exception as e:
        print(f"💥 Error: {e}")

def main():
    print("🧪 Google Scholar Title-Only Search Test")
    print("=" * 40)
    print("Testing new behavior: title-only search, extract from results")
    print()
    
    while True:
        print("Choose test:")
        print("1. Test with known papers")
        print("2. Test with DBLP papers")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            test_title_only_search()
        elif choice == "2":
            test_dblp_papers()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()