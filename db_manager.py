#!/usr/bin/env python3
"""
Database management CLI tool for academic papers.
Provides utilities for managing, exporting, and maintaining the papers database.
"""
import argparse
import json
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import pandas as pd

from database import PaperDatabase

console = Console()

def cmd_stats(db: PaperDatabase, args):
    """Display database statistics."""
    stats = db.get_statistics()
    datasets = db.get_datasets()
    
    console.print(Panel.fit(
        "[bold blue]Database Statistics[/bold blue]",
        border_style="blue"
    ))
    
    # Overall stats
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Total Papers", str(stats['total_papers']))
    table.add_row("Total Datasets", str(stats['total_datasets']))
    table.add_row("Papers with AI Summary", str(stats['papers_with_summary']))
    table.add_row("Papers with Keywords", str(stats['papers_with_keywords']))
    
    console.print(table)
    
    # Dataset breakdown
    if datasets:
        console.print("\n[bold blue]Datasets:[/bold blue]")
        
        dataset_table = Table(show_header=True, header_style="bold magenta")
        dataset_table.add_column("Name", style="cyan")
        dataset_table.add_column("Papers", style="yellow")
        dataset_table.add_column("Description", style="green")
        dataset_table.add_column("Source File", style="blue")
        dataset_table.add_column("Created", style="magenta")
        
        for dataset in datasets:
            created = dataset['created_at'].split('T')[0] if dataset['created_at'] else 'Unknown'
            dataset_table.add_row(
                dataset['name'],
                str(dataset['actual_papers']),
                dataset['description'] or '-',
                os.path.basename(dataset['source_file']) if dataset['source_file'] else '-',
                created
            )
        
        console.print(dataset_table)

def cmd_list(db: PaperDatabase, args):
    """List papers with optional filtering."""
    if args.dataset:
        papers = db.get_papers_by_dataset(args.dataset)
        title = f"Papers from dataset: {args.dataset}"
    else:
        papers = db.get_all_papers()
        title = "All papers"
    
    if not papers:
        console.print("No papers found")
        return
    
    console.print(f"\n[bold blue]{title}[/bold blue] ({len(papers)} papers)")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Authors", style="yellow")
    table.add_column("Year", style="blue")
    table.add_column("Summary", style="magenta")
    
    for paper in papers[:args.limit if args.limit else len(papers)]:
        # Parse authors
        try:
            authors = json.loads(paper['authors']) if paper['authors'] else []
            authors_str = ', '.join(authors[:2])  # Show first 2 authors
            if len(authors) > 2:
                authors_str += f" +{len(authors) - 2}"
        except:
            authors_str = paper['authors'] or ''
        
        # Truncate long titles and summaries
        title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
        summary = (paper['summary'][:40] + "..." if paper['summary'] and len(paper['summary']) > 40 
                  else paper['summary'] or '-')
        
        table.add_row(
            str(paper['paper_id']),
            title,
            authors_str,
            str(paper['publication_year']) if paper['publication_year'] else '-',
            summary
        )
    
    console.print(table)
    
    if args.limit and len(papers) > args.limit:
        console.print(f"\n[yellow]Showing first {args.limit} of {len(papers)} papers[/yellow]")

def cmd_search(db: PaperDatabase, args):
    """Search papers by query."""
    results = db.search_papers(args.query, args.dataset)
    
    if not results:
        console.print(f"No papers found matching '{args.query}'")
        return
    
    console.print(f"\n[bold blue]Search Results[/bold blue] ({len(results)} papers matching '{args.query}')")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Match Context", style="yellow")
    table.add_column("Dataset", style="blue")
    
    for paper in results[:args.limit if args.limit else len(results)]:
        # Find matching context
        query_lower = args.query.lower()
        title = paper['title'] or ''
        abstract = paper['abstract'] or ''
        summary = paper['summary'] or ''
        
        if query_lower in title.lower():
            context = "Title match"
        elif query_lower in abstract.lower():
            context = "Abstract match"
        elif query_lower in summary.lower():
            context = "Summary match"
        else:
            context = "Keyword match"
        
        table.add_row(
            str(paper['paper_id']),
            (title[:50] + "..." if len(title) > 50 else title),
            context,
            paper['source_dataset']
        )
    
    console.print(table)

def cmd_export(db: PaperDatabase, args):
    """Export papers to various formats."""
    # Get papers
    if args.dataset:
        papers = db.get_papers_by_dataset(args.dataset)
        console.print(f"📊 Exporting papers from dataset: {args.dataset}")
    else:
        papers = db.get_all_papers()
        console.print("📊 Exporting all papers")
    
    if not papers:
        console.print("❌ No papers found to export")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format.lower() == 'csv':
        db.export_to_csv(str(output_path), args.dataset)
    
    elif args.format.lower() == 'json':
        # Convert to DataFrame for processing
        df = db.export_to_dataframe(args.dataset)
        
        # Create JSON export
        export_data = []
        for _, paper in df.iterrows():
            paper_dict = {
                'paper_id': paper.get('paper_id', ''),
                'title': paper.get('title', ''),
                'authors': paper.get('authors_parsed', []),
                'abstract': paper.get('abstract', ''),
                'summary': paper.get('summary', ''),
                'keywords': paper.get('keywords_parsed', []),
                'url': paper.get('url', ''),
                'doi': paper.get('doi', ''),
                'publication_year': paper.get('publication_year'),
                'venue': paper.get('venue', ''),
                'source_dataset': paper.get('source_dataset', ''),
                'created_at': paper.get('created_at', ''),
                'updated_at': paper.get('updated_at', '')
            }
            export_data.append(paper_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"✅ Exported {len(export_data)} papers to {output_path}")
    
    else:
        console.print(f"❌ Unsupported format: {args.format}")

def cmd_import_csv(db: PaperDatabase, args):
    """Import papers from CSV (basic import without AI analysis)."""
    if not os.path.exists(args.csv_file):
        console.print(f"❌ File not found: {args.csv_file}")
        return
    
    console.print(f"📥 Importing papers from {args.csv_file}")
    
    try:
        df = pd.read_csv(args.csv_file)
        console.print(f"📄 Loaded {len(df)} rows from CSV")
        
        # Show available columns
        console.print("📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            console.print(f"  {i}. {col}")
        
        # Basic column mapping (can be enhanced)
        dataset_name = args.dataset or Path(args.csv_file).stem
        db.get_or_create_dataset(dataset_name, f"Imported from {args.csv_file}", args.csv_file)
        
        added_count = 0
        for _, row in df.iterrows():
            # Simple mapping - adjust based on your CSV structure
            paper_data = {
                'paper_id': str(row.get('paper_id', row.get('id', f"import_{row.name}"))),
                'title': str(row.get('title', row.get('Title', ''))),
                'authors': [str(row.get('authors', row.get('Author', '')))],
                'abstract': str(row.get('abstract', row.get('Abstract', ''))),
                'summary': str(row.get('summary', '')),
                'keywords': str(row.get('keywords', '')).split(',') if row.get('keywords') else [],
                'url': str(row.get('url', row.get('URL', ''))),
                'doi': str(row.get('doi', row.get('DOI', ''))),
                'publication_year': int(row.get('year', 0)) if str(row.get('year', '')).isdigit() else None,
                'venue': str(row.get('venue', '')),
                'source_dataset': dataset_name
            }
            
            if db.add_paper(**paper_data):
                added_count += 1
        
        db.update_dataset_stats(dataset_name)
        console.print(f"✅ Successfully imported {added_count} papers to dataset '{dataset_name}'")
        
    except Exception as e:
        console.print(f"❌ Error importing CSV: {e}")

def cmd_cleanup(db: PaperDatabase, args):
    """Clean up database (remove duplicates, etc.)."""
    console.print("🧹 Cleaning up database...")
    
    # Remove duplicates
    removed = db.cleanup_duplicates()
    if removed > 0:
        console.print(f"✅ Removed {removed} duplicate papers")
    else:
        console.print("ℹ️  No duplicates found")
    
    # Update all dataset statistics
    datasets = db.get_datasets()
    for dataset in datasets:
        db.update_dataset_stats(dataset['name'])
    
    console.print("✅ Database cleanup complete")

def cmd_delete(db: PaperDatabase, args):
    """Delete papers or datasets."""
    if args.paper_id:
        # Delete specific paper
        paper = db.get_paper(args.paper_id)
        if not paper:
            console.print(f"❌ Paper not found: {args.paper_id}")
            return
        
        console.print(f"📄 Paper to delete: {paper['title']}")
        if Confirm.ask("Are you sure you want to delete this paper?"):
            # Note: We'd need to add a delete_paper method to the database class
            console.print("⚠️  Delete paper functionality not implemented yet")
    
    elif args.dataset:
        # Delete entire dataset
        papers = db.get_papers_by_dataset(args.dataset)
        if not papers:
            console.print(f"❌ Dataset not found: {args.dataset}")
            return
        
        console.print(f"📊 Dataset '{args.dataset}' contains {len(papers)} papers")
        if Confirm.ask("Are you sure you want to delete this entire dataset?"):
            # Note: We'd need to add delete methods to the database class
            console.print("⚠️  Delete dataset functionality not implemented yet")

def main():
    parser = argparse.ArgumentParser(description="Database management tool for academic papers")
    parser.add_argument('--database', default='papers.db', help='Database file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # List command  
    list_parser = subparsers.add_parser('list', help='List papers')
    list_parser.add_argument('--dataset', help='Filter by dataset')
    list_parser.add_argument('--limit', type=int, default=20, help='Limit number of results')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search papers')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--dataset', help='Filter by dataset')
    search_parser.add_argument('--limit', type=int, default=10, help='Limit number of results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export papers')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')
    export_parser.add_argument('--dataset', help='Filter by dataset')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import papers from CSV')
    import_parser.add_argument('csv_file', help='CSV file to import')
    import_parser.add_argument('--dataset', help='Dataset name (default: filename)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up database')
    
    # Delete command (for future implementation)
    delete_parser = subparsers.add_parser('delete', help='Delete papers or datasets')
    delete_parser.add_argument('--paper-id', help='Paper ID to delete')
    delete_parser.add_argument('--dataset', help='Dataset to delete')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check if database exists for most commands
    if args.command != 'import' and not os.path.exists(args.database):
        console.print(f"❌ Database not found: {args.database}")
        console.print("💡 Use 'import' command or run parse_paper_db.py to create database")
        return
    
    # Connect to database
    try:
        db = PaperDatabase(args.database)
        console.print(f"🗃️  Connected to database: {args.database}")
        
        # Execute command
        if args.command == 'stats':
            cmd_stats(db, args)
        elif args.command == 'list':
            cmd_list(db, args)
        elif args.command == 'search':
            cmd_search(db, args)
        elif args.command == 'export':
            cmd_export(db, args)
        elif args.command == 'import':
            cmd_import_csv(db, args)
        elif args.command == 'cleanup':
            cmd_cleanup(db, args)
        elif args.command == 'delete':
            cmd_delete(db, args)
        
    except Exception as e:
        console.print(f"❌ Database error: {e}")
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main()