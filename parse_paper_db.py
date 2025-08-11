#!/usr/bin/env python3
"""
Enhanced paper parser that stores results in SQLite database.
Workflow: CSV file -> OpenAI analysis -> SQLite database
"""

import json
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
from src.openai_api import get_openai_client, get_openai_response
from database import PaperDatabase

console = Console()

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_config(config: dict, config_path: str = "config.json"):
    """Save configuration to JSON file."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(config_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def get_api_credentials(config: dict) -> tuple:
    """Get OpenAI API credentials from config or user input."""
    api_key = config.get('openai_api_key', '')
    base_url = config.get('openai_base_url', '')
    
    if not api_key:
        api_key = console.input("Enter your OpenAI API key: ").strip()
        
    if not base_url:
        base_url = console.input(
            "Enter the OpenAI API base URL (default: https://api.openai.com/v1): "
        ).strip() or "https://api.openai.com/v1"
    
    return api_key, base_url

def parse_csv_paper_data(file_path: str) -> pd.DataFrame:
    """Parse CSV file and extract paper information."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            console.print("❌ The provided CSV file is empty.")
            return None
        
        console.print(f"📄 Loaded {len(df)} papers from {file_path}")
        
        # Display column information to help user understand structure
        console.print("\n📋 Available columns:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Column Name", style="cyan", no_wrap=True)
        table.add_column("Sample Value", style="green")
        table.add_column("Non-null Count", style="yellow")
        
        for col in df.columns:
            sample_val = str(df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A")
            if len(sample_val) > 50:
                sample_val = sample_val[:47] + "..."
            non_null_count = df[col].notna().sum()
            table.add_row(col, sample_val, str(non_null_count))
        
        console.print(table)
        return df
        
    except Exception as e:
        console.print(f"❌ Error loading CSV file: {e}")
        return None

def map_csv_columns(df: pd.DataFrame) -> dict:
    """Map CSV columns to standard paper fields."""
    console.print("\n🔄 Column Mapping Configuration")
    
    # Common column mappings for different formats
    common_mappings = {
        'title': ['title', 'Title', 'paper_title', 'document_title', 'name'],
        'authors': ['authors', 'Author', 'author', 'Authors', 'creator'],
        'abstract': ['abstract', 'Abstract', 'Abstract Note', 'summary', 'description'],
        'url': ['url', 'URL', 'Url', 'link', 'Link', 'pdf_link'],
        'doi': ['doi', 'DOI', 'Doi'],
        'year': ['year', 'Year', 'Publication Year', 'publication_year', 'date'],
        'venue': ['venue', 'Venue', 'Publication Title', 'journal', 'conference'],
        'paper_id': ['id', 'ID', 'Key', 'key', 'paper_id', 'doi', 'DOI']
    }
    
    mapping = {}
    columns = df.columns.tolist()
    
    for field, possible_cols in common_mappings.items():
        found = False
        for possible_col in possible_cols:
            if possible_col in columns:
                mapping[field] = possible_col
                console.print(f"✓ {field} -> {possible_col}")
                found = True
                break
        
        if not found:
            console.print(f"⚠️  {field} column not found automatically")
            user_input = console.input(f"Enter column name for {field} (or press Enter to skip): ").strip()
            if user_input and user_input in columns:
                mapping[field] = user_input
                console.print(f"✓ {field} -> {user_input}")
    
    return mapping

def extract_paper_info(row: pd.Series, column_mapping: dict) -> dict:
    """Extract paper information from CSV row using column mapping."""
    paper_info = {}
    
    # Required fields
    paper_info['title'] = row.get(column_mapping.get('title', ''), '').strip()
    paper_info['abstract'] = row.get(column_mapping.get('abstract', ''), '').strip()
    
    # Optional fields
    paper_info['paper_id'] = str(row.get(column_mapping.get('paper_id', ''), 
                                       f"paper_{row.name}")).strip()
    
    authors_str = row.get(column_mapping.get('authors', ''), '')
    if authors_str and isinstance(authors_str, str):
        # Split authors by common delimiters
        authors = [a.strip() for a in authors_str.replace(';', ',').split(',') if a.strip()]
        paper_info['authors'] = authors
    else:
        paper_info['authors'] = []
    
    paper_info['url'] = row.get(column_mapping.get('url', ''), '').strip()
    paper_info['doi'] = row.get(column_mapping.get('doi', ''), '').strip()
    paper_info['venue'] = row.get(column_mapping.get('venue', ''), '').strip()
    
    # Handle year
    year_val = row.get(column_mapping.get('year', ''), '')
    if year_val and str(year_val).isdigit():
        paper_info['publication_year'] = int(year_val)
    else:
        paper_info['publication_year'] = None
    
    return paper_info

def analyze_paper_with_ai(paper_info: dict, openai_client, model: str = "gpt-4o-mini") -> dict:
    """Analyze paper abstract using OpenAI API."""
    abstract = paper_info.get('abstract', '')
    if not abstract:
        return {'summary': '', 'keywords': []}
    
    response = get_openai_response(openai_client, abstract, model=model)
    if not response:
        return {'summary': '', 'keywords': []}
    
    try:
        result = json.loads(response)
        return {
            'summary': result.get('summary', ''),
            'keywords': result.get('keywords', [])
        }
    except json.JSONDecodeError:
        console.print(f"⚠️ Failed to parse AI response for paper: {paper_info.get('title', 'Unknown')}")
        return {'summary': '', 'keywords': []}

def process_papers_to_database(df: pd.DataFrame, 
                             column_mapping: dict, 
                             db: PaperDatabase,
                             dataset_name: str,
                             openai_client,
                             model: str = "gpt-4o-mini",
                             skip_existing: bool = True) -> dict:
    """Process papers and store them in the database."""
    
    stats = {
        'total': len(df),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'ai_analyzed': 0
    }
    
    console.print(f"\n🚀 Processing {stats['total']} papers...")
    console.print(f"📊 Dataset: {dataset_name}")
    console.print(f"🤖 AI Model: {model}")
    
    for idx, row in track(df.iterrows(), total=len(df), description="Processing papers"):
        try:
            # Extract paper information
            paper_info = extract_paper_info(row, column_mapping)
            
            if not paper_info['title']:
                console.print(f"⚠️ Skipping row {idx}: No title found")
                stats['skipped'] += 1
                continue
            
            # Check if paper already exists
            if skip_existing and db.get_paper(paper_info['paper_id']):
                stats['skipped'] += 1
                continue
            
            # Analyze with AI if abstract is available
            ai_result = {'summary': '', 'keywords': []}
            if paper_info['abstract'] and openai_client:
                ai_result = analyze_paper_with_ai(paper_info, openai_client, model)
                if ai_result['summary'] or ai_result['keywords']:
                    stats['ai_analyzed'] += 1
            
            # Add to database
            paper_info.update({
                'summary': ai_result['summary'],
                'keywords': ai_result['keywords'],
                'source_dataset': dataset_name
            })
            
            paper_id = db.add_paper(**paper_info)
            if paper_id:
                stats['processed'] += 1
            else:
                stats['skipped'] += 1  # Duplicate or invalid
                
        except Exception as e:
            console.print(f"❌ Error processing row {idx}: {e}")
            stats['errors'] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Parse papers from CSV and store in database")
    parser.add_argument('csv_file', help="Path to CSV file containing paper metadata")
    parser.add_argument('--database', default="papers.db", help="Database file path")
    parser.add_argument('--dataset-name', help="Name for this dataset (default: filename)")
    parser.add_argument('--model', default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument('--config', default="config.json", help="Configuration file path")
    parser.add_argument('--skip-existing', action='store_true', help="Skip papers that already exist")
    parser.add_argument('--no-ai', action='store_true', help="Skip AI analysis")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        console.print(f"❌ File not found: {args.csv_file}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get dataset name
    dataset_name = args.dataset_name or Path(args.csv_file).stem
    
    # Display banner
    console.print(Panel.fit(
        f"[bold blue]Paper Parser with Database Storage[/bold blue]\n"
        f"📁 Input: {args.csv_file}\n"
        f"🗃️  Database: {args.database}\n"
        f"📊 Dataset: {dataset_name}",
        border_style="blue"
    ))
    
    # Parse CSV file
    df = parse_csv_paper_data(args.csv_file)
    if df is None:
        sys.exit(1)
    
    # Map columns
    column_mapping = map_csv_columns(df)
    
    # Initialize OpenAI client if AI analysis is enabled
    openai_client = None
    if not args.no_ai:
        try:
            api_key, base_url = get_api_credentials(config)
            
            # Save credentials to config
            config.update({
                'openai_api_key': api_key,
                'openai_base_url': base_url
            })
            save_config(config, args.config)
            
            openai_client = get_openai_client(api_key, base_url)
            console.print(f"🤖 OpenAI client initialized with base URL: {base_url}")
            
        except Exception as e:
            console.print(f"⚠️ Failed to initialize OpenAI client: {e}")
            console.print("Proceeding without AI analysis...")
    
    # Initialize database
    try:
        db = PaperDatabase(args.database)
        console.print(f"🗃️  Connected to database: {args.database}")
        
        # Create/update dataset
        db.get_or_create_dataset(
            dataset_name,
            f"Papers from {os.path.basename(args.csv_file)}",
            args.csv_file
        )
        
    except Exception as e:
        console.print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)
    
    try:
        # Process papers
        stats = process_papers_to_database(
            df, column_mapping, db, dataset_name, 
            openai_client, args.model, args.skip_existing
        )
        
        # Update dataset statistics
        db.update_dataset_stats(dataset_name)
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]Processing Complete![/bold green]")
        
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Count", style="yellow")
        
        results_table.add_row("Total papers", str(stats['total']))
        results_table.add_row("Successfully processed", str(stats['processed']))
        results_table.add_row("Skipped (existing/invalid)", str(stats['skipped']))
        results_table.add_row("Errors", str(stats['errors']))
        results_table.add_row("AI analyzed", str(stats['ai_analyzed']))
        
        console.print(results_table)
        
        # Display database statistics
        db_stats = db.get_statistics()
        console.print(f"\n📊 Database now contains {db_stats['total_papers']} total papers across {db_stats['total_datasets']} datasets")
        
        # Show export options
        console.print(f"\n💡 Next steps:")
        console.print(f"   • Run clustering: uv run cluster_from_db.py --dataset '{dataset_name}'")
        console.print(f"   • Export to CSV: python database.py export --dataset '{dataset_name}' --output papers.csv")
        console.print(f"   • View in browser: uv run serve_generic.py")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        console.print(f"❌ Error during processing: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()