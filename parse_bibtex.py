#!/usr/bin/env python3
"""
BibTeX file processing module for academic papers.
Parses BibTeX entries and extracts paper information for database storage.
"""

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TaskID
from loguru import logger

from database import PaperDatabase

console = Console()

def clean_bibtex_field(field_value: str) -> str:
    """Clean and normalize BibTeX field values."""
    if not field_value:
        return ""
    
    # Remove LaTeX commands and braces
    cleaned = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', field_value)
    cleaned = re.sub(r'[{}]', '', cleaned)
    
    # Replace LaTeX special characters
    replacements = {
        r'\\&': '&',
        r'\\%': '%',
        r'\\$': '$',
        r'\\_': '_',
        r'\\#': '#',
        r'\\textquotedblleft': '"',
        r'\\textquotedblright': '"',
        r'\\textquoteleft': "'",
        r'\\textquoteright': "'",
        r'``': '"',
        r"''": '"',
        r'`': "'",
        r'--': '–',
        r'---': '—',
        r'\\ ': ' ',
        r'\\,': ' ',
        r'\\;': ' ',
        r'\\:': ' ',
    }
    
    for latex_cmd, replacement in replacements.items():
        cleaned = re.sub(latex_cmd, replacement, cleaned)
    
    # Clean up whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def extract_authors_from_bibtex(author_field: str) -> List[str]:
    """Extract and parse author names from BibTeX author field."""
    if not author_field:
        return []
    
    # Clean the author field first
    cleaned_authors = clean_bibtex_field(author_field)
    
    # Split authors by 'and' (BibTeX standard)
    authors = []
    parts = re.split(r'\s+and\s+', cleaned_authors)
    
    for author in parts:
        author = author.strip()
        if not author:
            continue
            
        # Handle different name formats
        if ',' in author:
            # Format: "Last, First Middle"
            parts = author.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_names = parts[1].strip()
                if first_names:
                    formatted_name = f"{first_names} {last_name}"
                else:
                    formatted_name = last_name
            else:
                formatted_name = author
        else:
            # Format: "First Middle Last"
            formatted_name = author
            
        authors.append(formatted_name)
    
    return authors

def extract_keywords_from_bibtex(keywords_field: str) -> List[str]:
    """Extract keywords from BibTeX keywords field."""
    if not keywords_field:
        return []
    
    cleaned_keywords = clean_bibtex_field(keywords_field)
    
    # Split keywords by common separators
    keywords = []
    separators = [';', ',', '\n', '\r\n']
    
    current_keywords = [cleaned_keywords]
    for separator in separators:
        new_keywords = []
        for keyword_group in current_keywords:
            new_keywords.extend(keyword_group.split(separator))
        current_keywords = new_keywords
    
    # Clean and filter keywords
    for keyword in current_keywords:
        keyword = keyword.strip()
        if keyword and len(keyword) > 1:
            keywords.append(keyword)
    
    return keywords

def extract_year_from_bibtex(entry: Dict[str, Any]) -> Optional[int]:
    """Extract publication year from BibTeX entry."""
    # Try different year fields
    year_fields = ['year', 'date', 'urldate']
    
    for field in year_fields:
        if field in entry:
            year_str = clean_bibtex_field(entry[field])
            # Extract 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if year_match:
                return int(year_match.group())
    
    return None

def bibtex_entry_to_paper_data(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a BibTeX entry to paper data format."""
    # Extract basic fields
    title = clean_bibtex_field(entry.get('title', ''))
    authors_list = extract_authors_from_bibtex(entry.get('author', ''))
    abstract = clean_bibtex_field(entry.get('abstract', ''))
    keywords_list = extract_keywords_from_bibtex(entry.get('keywords', ''))
    
    # Extract publication details
    year = extract_year_from_bibtex(entry)
    venue = clean_bibtex_field(entry.get('journal', entry.get('booktitle', entry.get('publisher', ''))))
    doi = clean_bibtex_field(entry.get('doi', ''))
    url = clean_bibtex_field(entry.get('url', ''))
    
    # Generate paper ID
    paper_id = entry.get('ID', '')
    if not paper_id:
        # Generate ID from title or DOI
        if doi:
            paper_id = doi
        elif title:
            paper_id = hashlib.md5(title.encode('utf-8')).hexdigest()[:8]
        else:
            paper_id = hashlib.md5(str(entry).encode('utf-8')).hexdigest()[:8]
    
    return {
        'paper_id': paper_id,
        'title': title,
        'authors': authors_list,
        'abstract': abstract,
        'keywords': keywords_list,
        'url': url,
        'doi': doi,
        'publication_year': year,
        'venue': venue,
        'entry_type': entry.get('ENTRYTYPE', 'unknown')
    }

def parse_bibtex_file(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a BibTeX file and return paper data and statistics."""
    console.print(f"📚 Parsing BibTeX file: {file_path}")
    
    # Configure parser
    parser = BibTexParser()
    parser.customization = convert_to_unicode
    
    try:
        with open(file_path, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file, parser=parser)
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as bibtex_file:
                    bib_database = bibtexparser.load(bibtex_file, parser=parser)
                console.print(f"✅ Successfully parsed with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not parse BibTeX file with any common encoding")
    
    console.print(f"📖 Found {len(bib_database.entries)} BibTeX entries")
    
    # Parse entries
    papers_data = []
    stats = {
        'total_entries': len(bib_database.entries),
        'processed_papers': 0,
        'skipped_entries': 0,
        'errors': 0,
        'error_details': []
    }
    
    with Progress() as progress:
        task = progress.add_task("Processing BibTeX entries...", total=len(bib_database.entries))
        
        for i, entry in enumerate(bib_database.entries):
            try:
                # Convert entry to paper data
                paper_data = bibtex_entry_to_paper_data(entry)
                
                # Validate essential fields
                if not paper_data['title']:
                    stats['skipped_entries'] += 1
                    stats['error_details'].append(f"Entry {i+1}: No title found")
                    continue
                
                papers_data.append(paper_data)
                stats['processed_papers'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                stats['error_details'].append(f"Entry {i+1}: {str(e)}")
                logger.error(f"Error processing BibTeX entry {i+1}: {e}")
                
            progress.update(task, advance=1)
    
    console.print(f"✅ Successfully processed {stats['processed_papers']} papers")
    if stats['skipped_entries'] > 0:
        console.print(f"⚠️  Skipped {stats['skipped_entries']} entries (no title)")
    if stats['errors'] > 0:
        console.print(f"❌ Errors in {stats['errors']} entries")
    
    return papers_data, stats

def process_bibtex_file(
    file_path: str,
    db: PaperDatabase,
    dataset_name: str,
    description: str = "",
    dry_run: bool = False,
    upload_to_existing: bool = False
) -> Dict[str, Any]:
    """Process a BibTeX file and add papers to the database."""
    
    console.print(f"🔄 Processing BibTeX file: {Path(file_path).name}")
    
    # Parse the BibTeX file
    papers_data, parse_stats = parse_bibtex_file(file_path)
    
    if not papers_data:
        return {
            'message': 'No valid papers found in BibTeX file',
            'statistics': parse_stats
        }
    
    # Process papers
    final_stats = {
        **parse_stats,
        'added_papers': 0,
        'duplicate_papers': 0,
        'final_errors': 0
    }
    
    if dry_run:
        console.print("🧪 Dry run mode - no papers will be added to database")
        return {
            'message': f'Dry run completed. Would process {len(papers_data)} papers.',
            'statistics': final_stats,
            'sample_papers': papers_data[:5]  # Return first 5 as sample
        }
    
    # Create or verify dataset
    if not upload_to_existing:
        dataset_id = db.create_dataset(dataset_name, description)
        console.print(f"📁 Created new dataset: {dataset_name}")
    else:
        # Verify dataset exists
        datasets = {d['name']: d for d in db.get_datasets()}
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        console.print(f"📁 Adding to existing dataset: {dataset_name}")
    
    # Add papers to database
    console.print("💾 Adding papers to database...")
    
    with Progress() as progress:
        task = progress.add_task("Adding papers...", total=len(papers_data))
        
        for paper_data in papers_data:
            try:
                if upload_to_existing:
                    # Use enhanced deduplication for existing datasets
                    result = db.add_paper_with_deduplication(
                        paper_id=paper_data['paper_id'],
                        title=paper_data['title'],
                        authors=paper_data['authors'],
                        abstract=paper_data['abstract'],
                        url=paper_data['url'],
                        doi=paper_data['doi'],
                        publication_year=paper_data['publication_year'],
                        venue=paper_data['venue'],
                        source_dataset=dataset_name,
                        keywords=paper_data['keywords']
                    )
                    
                    if result['status'] == 'added':
                        final_stats['added_papers'] += 1
                    elif result['status'] == 'duplicate':
                        final_stats['duplicate_papers'] += 1
                    else:
                        final_stats['final_errors'] += 1
                else:
                    # Standard addition for new datasets
                    paper_id = db.add_paper(
                        paper_id=paper_data['paper_id'],
                        title=paper_data['title'],
                        authors=paper_data['authors'],
                        abstract=paper_data['abstract'],
                        url=paper_data['url'],
                        doi=paper_data['doi'],
                        publication_year=paper_data['publication_year'],
                        venue=paper_data['venue'],
                        source_dataset=dataset_name,
                        keywords=paper_data['keywords']
                    )
                    final_stats['added_papers'] += 1
                    
            except Exception as e:
                final_stats['final_errors'] += 1
                final_stats['error_details'].append(f"Database error for '{paper_data['title']}': {str(e)}")
                logger.error(f"Error adding paper to database: {e}")
                
            progress.update(task, advance=1)
    
    # Update dataset statistics
    db.update_dataset_stats(dataset_name)
    
    console.print(f"✅ Processing complete!")
    console.print(f"   📄 Added: {final_stats['added_papers']} papers")
    if final_stats['duplicate_papers'] > 0:
        console.print(f"   🔄 Duplicates skipped: {final_stats['duplicate_papers']} papers")
    if final_stats['final_errors'] > 0:
        console.print(f"   ❌ Errors: {final_stats['final_errors']} papers")
    
    return {
        'message': f'Successfully processed BibTeX file. Added {final_stats["added_papers"]} papers to dataset "{dataset_name}".',
        'statistics': final_stats
    }

def main():
    """Command-line interface for BibTeX processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process BibTeX files and add papers to database")
    parser.add_argument('bibtex_file', help='Path to BibTeX file')
    parser.add_argument('--database', default='papers.db', help='Database file path')
    parser.add_argument('--dataset', required=True, help='Dataset name for papers')
    parser.add_argument('--description', default='', help='Dataset description')
    parser.add_argument('--dry-run', action='store_true', help='Parse file but don\'t add to database')
    parser.add_argument('--existing', action='store_true', help='Add to existing dataset (enable deduplication)')
    
    args = parser.parse_args()
    
    # Check if BibTeX file exists
    if not Path(args.bibtex_file).exists():
        console.print(f"❌ BibTeX file not found: {args.bibtex_file}")
        return
    
    # Initialize database
    db = PaperDatabase(args.database)
    
    try:
        result = process_bibtex_file(
            args.bibtex_file,
            db,
            args.dataset,
            args.description,
            args.dry_run,
            args.existing
        )
        
        console.print(f"\n[bold green]{result['message']}[/bold green]")
        
        if 'sample_papers' in result:
            console.print("\n[bold blue]Sample papers that would be processed:[/bold blue]")
            for i, paper in enumerate(result['sample_papers'], 1):
                console.print(f"  {i}. {paper['title']}")
                console.print(f"     Authors: {', '.join(paper['authors'])}")
                console.print(f"     Year: {paper['publication_year'] or 'N/A'}")
                console.print()
        
    except Exception as e:
        console.print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()