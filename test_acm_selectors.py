#!/usr/bin/env python3
"""
Test script to verify ACM Digital Library abstract extraction with the new selectors.
"""

from src.abstract_extractor import AbstractExtractor
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_acm_selectors():
    """Test the ACM selectors on a sample ACM paper."""
    
    # Example ACM URL (replace with actual ACM paper URL)
    # This should be a URL from your DBLP results that redirects to ACM
    test_url = "https://doi.org/10.1145/3503222.3507771"  # Example DOI that should redirect to ACM
    
    print("🧪 Testing ACM Abstract Extraction")
    print("=" * 40)
    print(f"URL: {test_url}")
    
    print(f"\n🖥️  Opening browser (visible mode for debugging)...")
    extractor = AbstractExtractor(headless=False, debug=True)
    
    try:
        print(f"\n🚀 Starting extraction...")
        abstract = extractor.extract_abstract(test_url)
        
        if abstract:
            print(f"\n✅ SUCCESS!")
            print(f"Abstract length: {len(abstract)} characters")
            print(f"\n📄 Full Abstract:")
            print("-" * 40)
            print(abstract)
            print("-" * 40)
        else:
            print(f"\n❌ FAILED to extract abstract")
            print("Check the browser window and debug output for details")
    
    except Exception as e:
        print(f"\n💥 ERROR: {str(e)}")
    
    finally:
        input("\n⏸️  Check the browser window. Press Enter to close...")
        extractor._teardown_driver()

def test_acm_selectors_manual():
    """Manual test with an ACM URL you provide."""
    
    print("🧪 Manual ACM Abstract Extraction Test")
    print("=" * 40)
    
    acm_url = input("Enter an ACM Digital Library URL to test: ").strip()
    
    if not acm_url:
        print("❌ No URL provided!")
        return
    
    print(f"\n🖥️  Opening browser (visible mode)...")
    extractor = AbstractExtractor(headless=False, debug=True)
    
    try:
        print(f"\n🚀 Testing extraction on: {acm_url}")
        abstract = extractor.extract_abstract(acm_url)
        
        if abstract:
            print(f"\n✅ SUCCESS!")
            print(f"Abstract length: {len(abstract)} characters")
            print(f"\n📄 Extracted Abstract:")
            print("-" * 40)
            # Show in chunks for readability
            for i in range(0, len(abstract), 80):
                print(abstract[i:i+80])
            print("-" * 40)
        else:
            print(f"\n❌ FAILED to extract abstract")
    
    except Exception as e:
        print(f"\n💥 ERROR: {str(e)}")
    
    finally:
        input("\n⏸️  Press Enter to close browser...")
        extractor._teardown_driver()

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test with example URL")
    print("2. Test with your own ACM URL")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_acm_selectors()
    elif choice == "2":
        test_acm_selectors_manual()
    else:
        print("Invalid choice!")