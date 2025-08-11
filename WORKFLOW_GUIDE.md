# Complete Workflow: CSV to Visualization

This document describes the complete workflow from input CSV file to interactive visualization of clustered academic papers.

## Overview

The system processes CSV files exported from Zotero, stores them in a database, performs clustering analysis, and generates interactive visualizations. The workflow consists of 4 main steps:

```
CSV File → Database → Clustering → Visualization
```

## Prerequisites

- Python environment with `uv` package manager
- Required dependencies installed (automatically handled by `uv`)
- Web browser for viewing visualizations

## Step-by-Step Workflow

### Step 1: Process CSV to Database

**Input**: Zotero-exported CSV file (like `ASPLOS_labeled.csv`)
**Output**: SQLite database with papers

```bash
# Process Zotero CSV and add to database
uv run process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_2025" --description "ASPLOS 2025 Conference Papers" --force

# Verify the data was added
uv run db_manager.py stats
```

**What happens:**
- Parses CSV file with 89 columns (standard Zotero format + custom fields)
- Converts author format from "Last, First" to "First Last"
- Extracts keywords from multiple sources (keywords field, manual tags, automatic tags)
- Stores essential fields: title, authors, abstract, summary, keywords, DOI, URL, venue, year
- Creates dataset entry for organization

### Step 2: Generate Clustering Analysis

**Input**: Database with papers
**Output**: JSON file with clustering data for visualization

```bash
# Perform clustering analysis on the dataset
uv run cluster_from_db.py --database papers.db --dataset "ASPLOS_2025" --output-prefix asplos_analysis

# This creates: asplos_analysis_data.json
```

**What happens:**
- Loads papers from database for the specified dataset
- Applies TF-IDF vectorization on title + summary + keywords text
- Determines optimal number of clusters using silhouette analysis (max 15 clusters)
- Performs K-means clustering
- Applies PCA for 2D visualization coordinates
- Generates cluster analysis with top keywords and sample titles
- Exports to JSON format compatible with web visualization

### Step 3: View Interactive Visualization

**Input**: JSON clustering data
**Output**: Interactive web visualization

```bash
# Open the visualization in your browser
open clustering_visualization.html
```

**What happens:**
1. Web page loads with file upload interface
2. Upload the generated JSON file (`asplos_analysis_data.json`)
3. Interactive visualization displays with:
   - **Network Graph**: Papers as nodes, colored by cluster
   - **Cluster Panel**: Expandable cluster information
   - **Paper Details**: Click nodes for detailed paper information
   - **Statistics**: Total papers, clusters, and metadata

## File Structure

After running the complete workflow, you'll have:

```
paper-labeler/
├── papers.db                          # SQLite database with papers
├── asplos_analysis_data.json         # Clustering data for visualization
├── clustering_visualization.html      # Web interface for visualization
├── process_zotero_csv.py             # CSV to database processor
├── cluster_from_db.py                # Database to clustering analysis
└── db_manager.py                     # Database management utilities
```

## Quick Start Example

Here's a complete example using the ASPLOS dataset:

```bash
# 1. Process CSV to database
uv run process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_2025" --force

# 2. Generate clustering analysis
uv run cluster_from_db.py --database papers.db --dataset "ASPLOS_2025" --output-prefix asplos_viz

# 3. Open visualization
open clustering_visualization.html
# Then upload asplos_viz_data.json in the web interface
```

## Understanding the Visualization

### Network Graph
- **Nodes**: Each circle represents a paper
- **Colors**: Papers in the same cluster have the same color
- **Position**: 2D PCA coordinates show paper similarity
- **Size**: Node size can indicate importance or connectivity

### Cluster Information
- **Cluster panels**: Expandable sections showing cluster details
- **Top keywords**: Most characteristic terms for each cluster  
- **Sample titles**: Representative papers from each cluster
- **Paper count**: Number of papers in each cluster

### Interactive Features
- **Click nodes**: View detailed paper information
- **Hover effects**: Preview paper titles and authors
- **Cluster filtering**: Focus on specific research areas
- **Zoom and pan**: Explore different parts of the network

## Database Management

Monitor and manage your data:

```bash
# View database statistics
uv run db_manager.py stats

# List papers in a dataset
uv run db_manager.py list --dataset "ASPLOS_2025" --limit 10

# Search papers
uv run db_manager.py search "machine learning" --dataset "ASPLOS_2025"

# Export data
uv run db_manager.py export output.json --format json --dataset "ASPLOS_2025"
```

## Troubleshooting

### Common Issues

**1. CSV Processing Errors**
```bash
# Test with dry-run first
uv run process_zotero_csv.py your_file.csv --dataset "Test" --dry-run
```

**2. Empty Clustering Results**
- Ensure papers have sufficient text content (title, abstract, summary)
- Check that the dataset name matches exactly

**3. Visualization Not Loading**
- Verify the JSON file is valid
- Check browser console for errors
- Ensure the JSON file path is correct

**4. Database Issues**
```bash
# Check database integrity
uv run db_manager.py stats

# Clean up duplicates
uv run db_manager.py cleanup
```

## Advanced Usage

### Multiple Datasets
```bash
# Process multiple CSV files
uv run process_zotero_csv.py papers1.csv --dataset "Dataset1" --force
uv run process_zotero_csv.py papers2.csv --dataset "Dataset2" --force

# Cluster each dataset separately
uv run cluster_from_db.py --dataset "Dataset1" --output-prefix dataset1_viz
uv run cluster_from_db.py --dataset "Dataset2" --output-prefix dataset2_viz

# Or cluster all papers together
uv run cluster_from_db.py --output-prefix combined_viz
```

### Custom Clustering Parameters
```bash
# Specify maximum number of clusters
uv run cluster_from_db.py --dataset "ASPLOS_2025" --output-prefix custom --max-clusters 10

# Use different random seed for reproducibility
uv run cluster_from_db.py --dataset "ASPLOS_2025" --output-prefix repro --random-seed 123
```

## Output Files Description

### Database (papers.db)
- **papers table**: Individual paper records
- **datasets table**: Dataset metadata and organization
- **Normalized storage**: JSON fields for authors and keywords

### Clustering JSON
- **papers array**: Paper data with cluster assignments and PCA coordinates
- **cluster_info**: Detailed cluster analysis with keywords and statistics
- **metadata**: Clustering parameters and generation info

### Visualization HTML
- **Self-contained**: No external dependencies
- **Responsive design**: Works on desktop and mobile
- **Interactive features**: D3.js-based network visualization

This workflow provides a complete pipeline from raw academic paper data to interactive, clustered visualizations for research analysis and exploration.