from src.dblp_api import get_paper_list, get_candidcate_venues
from src.abstract_extractor import AbstractExtractor

venues = get_candidcate_venues("ASPLOS")

print("Top 10 venues:")
print(venues[:10])

# Get conference papers only (default behavior)
print(f"\n--- Getting Conference Papers Only ---")
data = get_paper_list("ASPLOS", 2023, filter_conference_papers=True)

print(f"\nFound {len(data)} conference papers. First 5 papers:")
for i, paper in enumerate(data[:5]):
    print(f"{i+1}. {paper.title}")
    print(f"   URL: {paper.url}")
    print(f"   Type: {paper.type}")

# Compare with all papers (including editorship, etc.)
print(f"\n--- Getting All Paper Types ---")
all_data = get_paper_list("ASPLOS", 2023, filter_conference_papers=False)
print(f"Found {len(all_data)} papers of all types vs {len(data)} conference papers")

# Extract abstracts using Google Scholar (much more reliable!)
print(f"\n--- Extracting Abstracts using Google Scholar ---")
print("📚 Using Google Scholar as primary source for better reliability")

extractor = AbstractExtractor(
    headless=True,  # Headless mode for background execution
    use_google_scholar=True,  # Enable Google Scholar
    scholar_mirror='mirror3'  # Use mirror3
)

papers_with_abstracts = extractor.extract_abstracts_batch(data[:3])  # Test first 3 papers

print("\nPapers with extracted abstracts:")
for i, paper in enumerate(papers_with_abstracts):
    print(f"\n{i+1}. {paper.title}")
    print(f"   Abstract: {paper.abstract[:200]}{'...' if len(paper.abstract) > 200 else ''}")