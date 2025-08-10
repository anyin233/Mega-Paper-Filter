#!/usr/bin/env python3
"""
Example script to check URL support for DBLP papers after redirect resolution.
"""

from src.dblp_api import get_paper_list
from src.abstract_extractor import AbstractExtractor

def main():
    # Get papers from ASPLOS 2023
    print("Fetching papers from ASPLOS 2023...")
    papers = get_paper_list("ASPLOS", 2023)
    
    if not papers:
        print("No papers found!")
        return
    
    print(f"Found {len(papers)} papers")
    
    # Check URL support for first 5 papers as example
    test_papers = papers[:5]
    print(f"\nChecking URL support for first {len(test_papers)} papers...")
    
    extractor = AbstractExtractor()  # Using default visible browser mode
    try:
        results = extractor.check_papers_url_support(test_papers)
        
        print(f"\n--- URL Support Results ---")
        print(f"Total papers checked: {results['total_papers']}")
        print(f"Supported papers: {results['supported_count']}")
        print(f"Unsupported papers: {results['unsupported_count']}")
        
        if results['unsupported_papers']:
            print(f"\nUnsupported papers:")
            for paper in results['unsupported_papers']:
                print(f"  • {paper['title'][:60]}{'...' if len(paper['title']) > 60 else ''}")
                print(f"    Original URL: {paper['original_url']}")
                print(f"    Final URL: {paper['final_url']}")
                print()
    
    except KeyboardInterrupt:
        print("\nCheck interrupted by user")
    except Exception as e:
        print(f"Error during check: {e}")
    finally:
        extractor._teardown_driver()

if __name__ == "__main__":
    main()