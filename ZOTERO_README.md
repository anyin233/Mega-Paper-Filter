# Zotero CSV Processor

This script processes CSV files exported from Zotero and adds the papers to the academic papers database. It's specifically designed to work with Zotero's standard CSV export format (89 columns including custom fields).

## Reference Format

The script is designed to work with CSV files like `ASPLOS_labeled.csv` which follows Zotero's export format with:
- 89 total columns (standard Zotero fields + custom `summary` and `keywords` fields)
- Standard Zotero metadata fields (Key, Item Type, Author, Title, etc.)
- Custom research fields (`summary`, `keywords`) added for enhanced paper analysis

## Features

- **Zotero Format Support**: Specifically designed for Zotero's CSV export format
- **Smart Field Mapping**: Automatically maps Zotero fields to database schema
- **Author Parsing**: Converts "Last, First" format to "First Last"
- **Keyword Extraction**: Supports multiple keyword sources (manual tags, automatic tags, keywords field)
- **Duplicate Detection**: Generates unique IDs to avoid duplicates
- **Data Validation**: Handles missing values and data type conversions
- **Preview Mode**: Dry-run capability to preview what would be processed
- **Progress Tracking**: Rich console output with progress bars

## Usage

```bash
# Process the reference ASPLOS dataset
python process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_2025" --description "ASPLOS 2025 conference papers"

# Basic usage - process any Zotero CSV
python process_zotero_csv.py papers.csv --dataset "MyPapers"

# With description
python process_zotero_csv.py papers.csv --dataset "Research_Papers" --description "My research paper collection"

# Dry run to preview processing
python process_zotero_csv.py papers.csv --dataset "Test" --dry-run

# Skip confirmations (useful for automation)
python process_zotero_csv.py papers.csv --dataset "Batch" --force

# Custom database location
python process_zotero_csv.py papers.csv --dataset "Papers" --database /path/to/custom.db
```

## Zotero Export Instructions

1. In Zotero, select the items you want to export
2. Right-click and choose "Export Items..."
3. Select "CSV" as the format
4. Check "Export Files" if you want file attachments info
5. Save the CSV file
6. Use this script to process the exported CSV

## Supported Item Types

The script processes the following Zotero item types:
- Conference Papers (`conferencePaper`)
- Journal Articles (`journalArticle`)

Other item types are filtered out automatically.

## Field Mapping

| Zotero Field | Database Field | Notes |
|--------------|---------------|-------|
| Title | title | Required field |
| Author | authors | Converted from "Last, First" to "First Last" |
| Abstract Note | abstract | Main abstract text |
| DOI | doi | Digital Object Identifier |
| Url | url | Paper URL |
| Publication Title | venue | Conference/Journal name |
| Publication Year | publication_year | Numeric year |
| summary | summary | Custom summary if available |
| keywords | keywords | Custom keywords if available |
| Manual Tags | keywords | Fallback if no keywords field |
| Automatic Tags | keywords | Additional fallback |

## Error Handling

The script includes robust error handling:
- Skips entries with missing titles
- Handles NaN/missing values gracefully
- Provides detailed error reports
- Continues processing even if individual entries fail

## Integration

This script integrates seamlessly with the existing paper database system:
- Uses the same database schema as `parse_paper_db.py`
- Compatible with `cluster_from_db.py` for clustering
- Works with `db_manager.py` for database management
- Supports the complete workflow: Zotero → Database → Clustering → Visualization