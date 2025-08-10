#!/usr/bin/env python3
"""
Example script to demonstrate DBLP paper type filtering.
"""

from src.dblp_api import get_paper_list, get_paper_type_statistics

def main():
    venue = "ASPLOS"
    year = 2023
    
    print(f"=== Paper Type Analysis for {venue} {year} ===\n")
    
    # Get type statistics
    print("Getting paper type statistics...")
    type_stats = get_paper_type_statistics(venue, year)
    
    print(f"\nPaper Types Found:")
    total_papers = sum(stats['count'] for stats in type_stats.values())
    
    for paper_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        percentage = (stats['count'] / total_papers) * 100
        print(f"\n📄 {paper_type}: {stats['count']} papers ({percentage:.1f}%)")
        for i, example in enumerate(stats['examples'], 1):
            print(f"   {i}. {example[:80]}{'...' if len(example) > 80 else ''}")
    
    print(f"\nTotal papers: {total_papers}")
    
    # Demonstrate filtering
    print(f"\n=== Filtering Demo ===")
    
    # Get only conference papers
    conference_papers = get_paper_list(venue, year, filter_conference_papers=True)
    
    # Get all papers
    all_papers = get_paper_list(venue, year, filter_conference_papers=False)
    
    print(f"All papers: {len(all_papers)}")
    print(f"Conference papers only: {len(conference_papers)}")
    print(f"Filtered out: {len(all_papers) - len(conference_papers)} papers")
    
    # Show first few conference papers
    print(f"\nFirst 5 Conference Papers:")
    for i, paper in enumerate(conference_papers[:5]):
        print(f"{i+1}. {paper.title}")
        print(f"   Type: {paper.type}")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print()

if __name__ == "__main__":
    main()