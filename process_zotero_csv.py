#!/usr/bin/env python3
"""
Process Zotero-exported CSV files and add papers to the database.
This script is specifically designed for Zotero's CSV export format.
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.prompt import Confirm
import hashlib

from database import PaperDatabase

console = Console()

def clean_author_list(author_str: str) -> List[str]:
    """Clean and parse author string from Zotero format."""
    if not author_str or pd.isna(author_str):
        return []
    
    # Zotero exports authors in format: "Last1, First1; Last2, First2"
    authors = []
    for author in author_str.split(';'):
        author = author.strip()
        if author:
            # Convert "Last, First" to "First Last"
            if ',' in author:
                parts = [p.strip() for p in author.split(',')]
                if len(parts) >= 2:
                    # Handle cases like "Last, First Middle" or "Last, F. M."
                    last_name = parts[0]
                    first_names = ' '.join(parts[1:])
                    author = f"{first_names} {last_name}".strip()
            authors.append(author)
    
    return authors

def clean_keywords_list(keywords_str: str) -> List[str]:
    """Clean and parse keywords string."""
    if not keywords_str or pd.isna(keywords_str) or str(keywords_str).lower() == 'nan':
        return []
    
    keywords_str = str(keywords_str)  # Ensure it's a string
    
    # Split by semicolon or comma and clean
    keywords = []
    separators = [';', ',']
    
    # Use the first separator found, or semicolon as default
    separator = ';'
    for sep in separators:
        if sep in keywords_str:
            separator = sep
            break
    
    for keyword in keywords_str.split(separator):
        keyword = keyword.strip()
        if keyword and keyword.lower() != 'nan':
            keywords.append(keyword)
    
    return keywords

def generate_paper_id(title: str, authors: List[str], year: Optional[int] = None) -> str:
    """Generate a unique paper ID based on title, authors, and year."""
    # Create a string to hash
    id_string = title.lower().strip()
    if authors:
        id_string += '|' + '|'.join(sorted([a.lower().strip() for a in authors]))
    if year:
        id_string += f'|{year}'
    
    # Generate short hash
    hash_obj = hashlib.md5(id_string.encode('utf-8'))
    return hash_obj.hexdigest()[:8].upper()

def process_zotero_csv(csv_path: str, db: PaperDatabase, dataset_name: str, 
                      description: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Process a Zotero-exported CSV file and add papers to the database.
    
    Args:
        csv_path: Path to the Zotero CSV file
        db: Database instance
        dataset_name: Name for the dataset
        description: Optional description for the dataset
        dry_run: If True, don't actually add papers to database
        
    Returns:
        Dictionary with processing statistics
    """
    console.print(f"📖 Reading Zotero CSV: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'utf-16']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                console.print(f"📝 Successfully read file with {encoding} encoding")
                break
            except:
                continue
        else:
            raise ValueError(f"Could not read CSV file with any common encoding")
    
    console.print(f"📊 Found {len(df)} entries in CSV")
    
    # Show the columns available
    console.print("📋 Available columns:")
    for i, col in enumerate(df.columns[:10], 1):  # Show first 10 columns
        console.print(f"  {i:2d}. {col}")
    if len(df.columns) > 10:
        console.print(f"  ... and {len(df.columns) - 10} more columns")
    
    # Filter for conference papers and journal articles
    if 'Item Type' in df.columns:
        valid_types = ['conferencePaper', 'journalArticle']
        df = df[df['Item Type'].isin(valid_types)]
        console.print(f"📑 Filtered to {len(df)} papers (conference papers and journal articles)")
    
    # Create dataset if not dry run
    if not dry_run:
        dataset_desc = description or f"Papers imported from Zotero CSV: {os.path.basename(csv_path)}"
        db.get_or_create_dataset(dataset_name, dataset_desc, csv_path)
    
    stats = {
        'total_entries': len(df),
        'processed': 0,
        'added': 0,
        'skipped': 0,
        'errors': 0,
        'error_details': []
    }
    
    with Progress() as progress:
        task = progress.add_task("[green]Processing papers...", total=len(df))
        
        for idx, row in df.iterrows():
            progress.update(task, advance=1)
            stats['processed'] += 1
            
            try:
                # Extract paper information from Zotero format
                title = str(row.get('Title', '') or '').strip()
                if not title:
                    stats['skipped'] += 1
                    continue
                
                # Parse authors
                authors = clean_author_list(row.get('Author', ''))
                
                # Extract year
                pub_year = None
                year_str = str(row.get('Publication Year', ''))
                if year_str and year_str.isdigit():
                    pub_year = int(year_str)
                
                # Generate paper ID
                paper_id = generate_paper_id(title, authors, pub_year)
                
                # Extract other fields (handle NaN values)
                abstract = str(row.get('Abstract Note', '') or '').strip()
                doi = str(row.get('DOI', '') or '').strip()
                url = str(row.get('Url', '') or '').strip()
                venue = str(row.get('Publication Title', '') or '').strip()
                
                # Handle existing summary and keywords from Zotero
                summary = str(row.get('summary', '') or '').strip()
                if summary.lower() == 'nan':
                    summary = ''
                keywords = clean_keywords_list(str(row.get('keywords', '') or ''))
                
                # If no keywords from 'keywords' column, try 'Manual Tags' or 'Automatic Tags'
                if not keywords:
                    manual_tags = clean_keywords_list(str(row.get('Manual Tags', '') or ''))
                    auto_tags = clean_keywords_list(str(row.get('Automatic Tags', '') or ''))
                    keywords = manual_tags + auto_tags
                
                paper_data = {
                    'paper_id': paper_id,
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'summary': summary,
                    'keywords': keywords,
                    'url': url,
                    'doi': doi,
                    'publication_year': pub_year,
                    'venue': venue,
                    'source_dataset': dataset_name
                }
                
                if not dry_run:
                    if db.add_paper(**paper_data):
                        stats['added'] += 1
                    else:
                        stats['skipped'] += 1
                else:
                    # In dry run, assume it would be added
                    stats['added'] += 1
                    
            except Exception as e:
                stats['errors'] += 1
                error_msg = f"Row {idx + 1}: {str(e)}"
                stats['error_details'].append(error_msg)
                console.print(f"❌ Error processing row {idx + 1}: {e}")
    
    # Update dataset statistics if not dry run
    if not dry_run and stats['added'] > 0:
        db.update_dataset_stats(dataset_name)
    
    return stats

def show_processing_summary(stats: Dict[str, Any], dataset_name: str, dry_run: bool = False):
    """Display processing summary."""
    mode = "DRY RUN - " if dry_run else ""
    
    summary_text = f"""
[bold green]{mode}Processing Complete![/bold green]

[cyan]Dataset:[/cyan] {dataset_name}
[cyan]Total entries in CSV:[/cyan] {stats['total_entries']}
[cyan]Processed:[/cyan] {stats['processed']}
[cyan]Successfully added:[/cyan] {stats['added']}
[cyan]Skipped:[/cyan] {stats['skipped']}
[cyan]Errors:[/cyan] {stats['errors']}
"""
    
    if stats['errors'] > 0:
        summary_text += "\n[red]Error Details:[/red]\n"
        for error in stats['error_details'][:5]:  # Show first 5 errors
            summary_text += f"  • {error}\n"
        if len(stats['error_details']) > 5:
            summary_text += f"  • ... and {len(stats['error_details']) - 5} more errors\n"
    
    console.print(Panel.fit(summary_text, border_style="green"))

def main():
    parser = argparse.ArgumentParser(
        description="Process Zotero-exported CSV files and add papers to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process ASPLOS papers from Zotero export (reference format)
  python process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_2025" --description "ASPLOS 2025 papers from Zotero"
  
  # Dry run to preview what would be processed
  python process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_Test" --dry-run
  
  # Process any Zotero CSV with custom database location
  python process_zotero_csv.py my_papers.csv --dataset "MyPapers" --database custom.db --force
        """
    )
    
    parser.add_argument('csv_file', help='Zotero-exported CSV file to process')
    parser.add_argument('--dataset', required=True, help='Dataset name for these papers')
    parser.add_argument('--database', default='papers.db', help='Database file path (default: papers.db)')
    parser.add_argument('--description', help='Description for the dataset')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without adding to database')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        console.print(f"❌ CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Show preview information
    console.print(Panel.fit(
        f"[bold blue]Zotero CSV Processor[/bold blue]\n\n"
        f"[cyan]CSV File:[/cyan] {args.csv_file}\n"
        f"[cyan]Dataset Name:[/cyan] {args.dataset}\n"
        f"[cyan]Database:[/cyan] {args.database}\n"
        f"[cyan]Description:[/cyan] {args.description or 'Auto-generated'}\n"
        f"[cyan]Mode:[/cyan] {'DRY RUN' if args.dry_run else 'LIVE PROCESSING'}",
        border_style="blue"
    ))
    
    # Confirmation (skip if dry run or force)
    if not args.dry_run and not args.force:
        if not Confirm.ask("Do you want to continue?"):
            console.print("❌ Processing cancelled")
            sys.exit(0)
    
    # Connect to database
    try:
        db = PaperDatabase(args.database)
        console.print(f"🗃️  Connected to database: {args.database}")
        
        # Check if dataset already exists
        if not args.dry_run:
            existing_papers = db.get_papers_by_dataset(args.dataset)
            if existing_papers:
                console.print(f"⚠️  Dataset '{args.dataset}' already has {len(existing_papers)} papers")
                if not args.force:
                    if not Confirm.ask("Do you want to continue? (This might add duplicates)"):
                        console.print("❌ Processing cancelled")
                        sys.exit(0)
        
        # Process the CSV
        stats = process_zotero_csv(
            args.csv_file, 
            db, 
            args.dataset,
            args.description,
            dry_run=args.dry_run
        )
        
        # Show summary
        show_processing_summary(stats, args.dataset, args.dry_run)
        
        if args.dry_run:
            console.print("\n💡 Run without --dry-run to actually add papers to the database")
        elif stats['added'] > 0:
            console.print(f"\n✅ Successfully added {stats['added']} papers to dataset '{args.dataset}'")
            console.print("📊 Use 'db_manager.py stats' to see updated database statistics")
            console.print("🔍 Use 'db_manager.py list --dataset {args.dataset}' to view the papers")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main()