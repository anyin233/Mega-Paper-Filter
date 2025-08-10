#!/usr/bin/env python3
"""
Configuration script to help users choose the best Google Scholar mirror.
"""

from src.google_scholar_extractor import GoogleScholarExtractor
import time

def test_all_mirrors():
    """Test all available mirrors and recommend the best one."""
    print("🌐 Testing All Google Scholar Mirrors")
    print("=" * 45)
    print("This will test connectivity to different Google Scholar mirrors...")
    print("Please wait, this may take a few minutes.\n")
    
    mirrors = {
        'default': 'https://scholar.google.com',
        'mirror1': 'https://scholar.google.ac.uk', 
        'mirror2': 'https://scholar.google.ca',
        'mirror3': 'https://scholar.google.com.au',
        'mirror4': 'https://scholar.google.de',
        'semantic': 'https://www.semanticscholar.org'
    }
    
    results = {}
    
    for mirror_name, url in mirrors.items():
        print(f"🔍 Testing {mirror_name}: {url}")
        
        try:
            start_time = time.time()
            
            with GoogleScholarExtractor(headless=True, preferred_mirror=mirror_name) as scholar:
                # Test with a well-known paper
                abstract = scholar.search_paper(
                    "Attention Is All You Need",
                    ["Ashish Vaswani"]
                )
                
                response_time = time.time() - start_time
                
                if abstract:
                    results[mirror_name] = {
                        'status': '✅ SUCCESS',
                        'response_time': response_time,
                        'abstract_length': len(abstract),
                        'url': url
                    }
                    print(f"   ✅ SUCCESS ({response_time:.2f}s) - Abstract: {len(abstract)} chars")
                else:
                    results[mirror_name] = {
                        'status': '⚠️  NO RESULTS',
                        'response_time': response_time,
                        'abstract_length': 0,
                        'url': url
                    }
                    print(f"   ⚠️  NO RESULTS ({response_time:.2f}s)")
                    
        except Exception as e:
            results[mirror_name] = {
                'status': '❌ FAILED',
                'response_time': 999,
                'error': str(e),
                'url': url
            }
            print(f"   ❌ FAILED: {str(e)[:50]}...")
        
        print()
    
    # Show summary and recommendation
    print("📊 SUMMARY")
    print("=" * 20)
    
    working_mirrors = [(name, data) for name, data in results.items() 
                      if data['status'] == '✅ SUCCESS']
    
    if working_mirrors:
        # Sort by response time
        working_mirrors.sort(key=lambda x: x[1]['response_time'])
        
        print("Working mirrors (ranked by speed):")
        for i, (name, data) in enumerate(working_mirrors, 1):
            print(f"{i}. {name}: {data['response_time']:.2f}s ({data['url']})")
        
        best_mirror = working_mirrors[0][0]
        print(f"\n🏆 RECOMMENDED: '{best_mirror}' (fastest working mirror)")
        print(f"To use this mirror, set: scholar_mirror='{best_mirror}'")
        
        # Save recommendation to config file
        try:
            with open('scholar_config.txt', 'w') as f:
                f.write(f"# Google Scholar Mirror Configuration\n")
                f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"RECOMMENDED_MIRROR={best_mirror}\n")
                f.write(f"RESPONSE_TIME={working_mirrors[0][1]['response_time']:.2f}s\n")
                f.write(f"\n# All working mirrors:\n")
                for name, data in working_mirrors:
                    f.write(f"{name}={data['response_time']:.2f}s\n")
            
            print("📄 Configuration saved to 'scholar_config.txt'")
        except Exception as e:
            print(f"⚠️  Could not save config: {e}")
    else:
        print("❌ NO WORKING MIRRORS FOUND!")
        print("This might indicate network connectivity issues or regional blocking.")
        print("Try running this test again later or check your internet connection.")

def interactive_test():
    """Interactive test to let user try specific mirrors."""
    mirrors = GoogleScholarExtractor.SCHOLAR_MIRRORS
    
    print("\n🧪 Interactive Mirror Test")
    print("=" * 25)
    
    print("Available mirrors:")
    for i, (name, url) in enumerate(mirrors.items(), 1):
        print(f"{i}. {name}: {url}")
    
    while True:
        try:
            choice = input(f"\nEnter mirror name to test (or 'quit'): ").strip()
            
            if choice.lower() == 'quit':
                break
            
            if choice not in mirrors:
                print(f"Unknown mirror '{choice}'. Available: {list(mirrors.keys())}")
                continue
            
            print(f"\n🔍 Testing {choice}...")
            
            with GoogleScholarExtractor(headless=False, preferred_mirror=choice) as scholar:
                title = input("Enter paper title to search: ").strip()
                if title:
                    abstract = scholar.search_paper(title)
                    if abstract:
                        print(f"✅ Found abstract ({len(abstract)} chars):")
                        print(f"{abstract[:200]}...")
                    else:
                        print("❌ No abstract found")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🌐 Google Scholar Mirror Configuration Tool")
    print("=" * 45)
    print("This tool helps you find the best Google Scholar mirror for your location.")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Test all mirrors automatically")
        print("2. Interactive mirror testing")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            test_all_mirrors()
        elif choice == "2":
            interactive_test()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()