from rich.console import Console

from src.dblp_api import get_paper_list, get_candidcate_venues
from src.abstract_extractor import AbstractExtractor


console = Console()

venue_name = console.input("Input the venue name to search for papers (e.g., 'ASPLOS'): ")

venues = get_candidcate_venues(venue_name)

console.print(f"\nFound {len(venues)} candidate venues for '{venue_name}':")
for i, venue in enumerate(venues, 1):
    console.print(f"{i}. [bold blue]{venue.name}[/bold blue] - [link={venue.url}]{venue.url}[/link] (Acronym: {venue.acronym})")

venue_choice = console.input("\nChoose a venue by number (or 'quit' to exit): ").strip()
if venue_choice.lower() == 'quit':
    console.print("👋 Goodbye!")
    exit()
try:
    venue_index = int(venue_choice) - 1
    if venue_index < 0 or venue_index >= len(venues):
        raise ValueError("Invalid venue number")
    selected_venue = venues[venue_index]
    console.print(f"\nYou selected: [bold blue]{selected_venue.name}[/bold blue] - [link={selected_venue.url}]{selected_venue.url}[/link] (Acronym: {selected_venue.acronym})")
except ValueError as e:
    console.print(f"❌ Error: {e}. Please enter a valid number.")

year = console.input("Enter the year to search for papers (e.g., 2023): ").strip()
try:
    year = int(year)
except ValueError:
    console.print("❌ Invalid year format. Please enter a valid integer.")
    exit()

# Get conference papers only (default behavior)
console.print(f"\n--- Getting Conference Papers📚 ---")
data = get_paper_list(selected_venue.acronym, year, filter_conference_papers=True)

console.print(f"\nFound {len(data)} conference papers.:")
# Display papers with pagination
papers_per_page = 5
total_papers = len(data)
total_pages = (total_papers + papers_per_page - 1) // papers_per_page
current_page = 1

while True:
    start_idx = (current_page - 1) * papers_per_page
    end_idx = min(start_idx + papers_per_page, total_papers)
    
    console.print(f"\n--- Page {current_page}/{total_pages} ---")
    for i in range(start_idx, end_idx):
        paper = data[i]
        console.print(f"{i+1}. [bold blue]{paper.title}[/bold blue]")
        console.print(f"   URL: [link={paper.url}]{paper.url}[/link]")
        console.print(f"   Type: {paper.type}")
    
    # Show navigation options
    nav_options = []
    if current_page > 1:
        nav_options.append("'p' for previous")
    if current_page < total_pages:
        nav_options.append("'n' for next")
    nav_options.append("'q' to continue")
    
    console.print(f"\nNavigation: {', '.join(nav_options)}")
    choice = console.input("Enter your choice: ").strip().lower()
    
    if choice == 'n' and current_page < total_pages:
        current_page += 1
    elif choice == 'p' and current_page > 1:
        current_page -= 1
    elif choice == 'q':
        break
    else:
        console.print("❌ Invalid choice. Please try again.")

# # Extract abstracts using Google Scholar (much more reliable!)
# print(f"\n--- Extracting Abstracts using Google Scholar ---")
# print("📚 Using Google Scholar as primary source for better reliability")

console.print("Choose the paper you want to extract abstracts for ")
paper_choice = console.input("Enter paper number(s) (e.g., 1,2,3,1-5 or all): ").strip()
try:
    if paper_choice.lower() == "all":
        selected_papers = data
    elif '-' in paper_choice:
        start, end = map(int, paper_choice.split('-'))
        selected_papers = data[start-1:end]
    else:
        indices = map(int, paper_choice.split(','))
        selected_papers = [data[i-1] for i in indices]
    
    if not selected_papers:
        raise ValueError("No valid papers selected")
except ValueError as e:
    console.print(f"❌ Error: {e}. Please enter valid paper numbers.")
    exit()
    

extractor = AbstractExtractor(
    headless=True,  # Headless mode for background execution
    use_google_scholar=True,  # Enable Google Scholar
    scholar_mirror='mirror3'  # Use mirror3
)

papers_with_abstracts = extractor.extract_abstracts_batch(selected_papers)

console.print(f"\nWriting abstracts to file...")
with open('papers_with_abstracts.txt', 'w') as f:
    for paper in papers_with_abstracts:
        f.write(f"{paper.title}\n")
        f.write(f"URL: {paper.url}\n")
        f.write(f"Type: {paper.type}\n")
        f.write(f"Abstract: {paper.abstract}\n\n")
console.print("✅ Abstracts written to 'papers_with_abstracts.txt'")

# print("\nPapers with extracted abstracts:")
for i, paper in enumerate(papers_with_abstracts[:3]):
    console.print(f"\n{i+1}. {paper.title}")
    console.print(f"   Abstract: {paper.abstract}")
    