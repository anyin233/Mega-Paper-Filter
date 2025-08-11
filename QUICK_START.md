# Quick Start Guide: CSV to Visualization

Get from CSV file to interactive visualization in 3 simple steps.

## 🚀 Quick 3-Step Process

### Step 1: CSV → Database
```bash
uv run process_zotero_csv.py ASPLOS_labeled.csv --dataset "ASPLOS_2025" --force
```
✅ **Result**: `papers.db` (SQLite database with your papers)

### Step 2: Database → Clustering
```bash
uv run cluster_from_db.py --database papers.db --output-prefix my_analysis
```
✅ **Result**: `my_analysis_data.json` (clustering data for visualization)

### Step 3: View Visualization
```bash
open clustering_visualization.html
```
Then upload `my_analysis_data.json` in the web interface.

✅ **Result**: Interactive network visualization of clustered papers

## 📁 Files Created

```
your-project/
├── papers.db                    # ← Step 1: Database with papers
├── my_analysis_data.json       # ← Step 2: Clustering data  
├── my_analysis_plots.png       # ← Step 2: Summary plots
└── clustering_visualization.html # ← Step 3: Web interface
```

## 🎯 What You Get

- **Interactive Network**: Papers as nodes, colored by research clusters
- **Cluster Analysis**: Top keywords and sample papers for each cluster  
- **Paper Details**: Click any node to see full paper information
- **Research Insights**: Discover patterns and relationships in your papers

## ⏱️ Typical Processing Time

- **72 ASPLOS papers**: ~30 seconds total
- **Step 1**: 5 seconds (CSV processing)
- **Step 2**: 20 seconds (clustering analysis)  
- **Step 3**: Instant (web visualization)

## 📊 Example Output

The ASPLOS dataset (72 papers) produces:
- **14 research clusters** (automatically determined)
- **Categories**: GPU computing, quantum circuits, memory systems, ML training, security, etc.
- **Visualization**: Interactive network with 72 nodes, color-coded by cluster

Ready to analyze your research papers? Start with Step 1!