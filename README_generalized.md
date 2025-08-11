# Academic Papers Clustering Visualization

A comprehensive, generalized system for clustering academic papers from any research domain and visualizing the results in an interactive web interface.

## 🎯 Features

### Universal Paper Clustering
- **Configurable CSV Input**: Works with any academic paper dataset format
- **Flexible Column Mapping**: Automatically adapt to different CSV structures
- **Advanced Text Processing**: TF-IDF vectorization with customizable parameters
- **Optimal Clustering**: Automatic cluster number detection using silhouette analysis
- **Multiple Formats**: Support for ASPLOS, ArXiv, IEEE, PubMed, and custom formats

### Interactive Web Visualization
- **File Upload Support**: Drag-and-drop JSON datasets directly into the browser
- **Dual View Modes**: 
  - Scatter plot with PCA dimensionality reduction
  - Interactive network graph with force-directed layout
- **Advanced Filtering**: Filter by clusters, search across all fields
- **Dynamic Interface**: Updates title and stats based on uploaded dataset
- **Responsive Design**: Works on desktop and mobile devices

### Comprehensive Analysis
- **Cluster Insights**: Top keywords, representative papers, and statistics
- **Visualization Options**: Matplotlib plots and interactive web charts
- **Export Capabilities**: JSON for web, CSV for analysis, PNG for presentations

## 🚀 Quick Start

### 1. Basic Usage
```bash
# Analyze any CSV file with default settings
uv run cluster_papers_generic.py your_papers.csv

# Use predefined configuration for common formats
uv run cluster_papers_generic.py papers.csv --config configs/arxiv_config.json

# Customize output location and prefix
uv run cluster_papers_generic.py papers.csv --output-dir ./results --output-prefix my_analysis
```

### 2. Launch Web Interface
```bash
# Start web server (automatically opens browser)
uv run serve_generic.py

# Use custom port and directory
uv run serve_generic.py --port 9000 --dir ./results

# Don't open browser automatically
uv run serve_generic.py --no-browser
```

### 3. Upload Custom Dataset
- Open the web interface
- Use the "Upload Dataset" button to load your JSON clustering results
- Or drag-and-drop a JSON file directly into the browser

## 📊 Supported Dataset Formats

### Predefined Configurations

#### ASPLOS Conference Papers
```bash
uv run cluster_papers_generic.py asplos.csv --config configs/asplos_config.json
```
Expected columns: `Title`, `Author`, `summary`, `keywords`, `Key`, `Publication Year`, `DOI`, `Url`

#### ArXiv Papers
```bash
uv run cluster_papers_generic.py arxiv.csv --config configs/arxiv_config.json
```
Expected columns: `title`, `authors`, `abstract`, `categories`, `id`, `year`, `doi`, `url`

#### IEEE Papers
```bash
uv run cluster_papers_generic.py ieee.csv --config configs/ieee_config.json
```
Expected columns: `Document Title`, `Authors`, `Abstract`, `Author Keywords`, `DOI`, `Publication Year`, `PDF Link`

#### PubMed Papers
```bash
uv run cluster_papers_generic.py pubmed.csv --config configs/pubmed_config.json
```
Expected columns: `Title`, `Authors`, `Abstract`, `Keywords`, `PMID`, `Publication Date`, `DOI`, `URL`

### Custom Column Mapping
```bash
# Map your CSV columns to the expected fields
uv run cluster_papers_generic.py custom.csv \
  --title-col "paper_title" \
  --author-col "paper_authors" \
  --summary-col "paper_abstract" \
  --keywords-col "paper_keywords" \
  --key-col "paper_id"
```

## ⚙️ Configuration

### Create Custom Configuration
```json
{
  "name": "Your Dataset Name",
  "description": "Description of your dataset",
  "csv_columns": {
    "title": "Title_Column_Name",
    "author": "Author_Column_Name", 
    "summary": "Abstract_Column_Name",
    "keywords": "Keywords_Column_Name",
    "key": "ID_Column_Name",
    "year": "Year_Column_Name",
    "doi": "DOI_Column_Name",
    "url": "URL_Column_Name"
  },
  "tfidf_params": {
    "max_features": 1000,
    "stop_words": "english",
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.8
  },
  "clustering_params": {
    "max_k": 15,
    "random_state": 42,
    "n_init": 10
  },
  "pca_params": {
    "n_components": 2,
    "random_state": 42
  }
}
```

### Advanced Parameters
- **max_features**: Maximum number of TF-IDF features (higher = more detailed)
- **ngram_range**: N-gram range for text analysis (e.g., [1,2] for unigrams and bigrams)
- **min_df**: Minimum document frequency (removes rare terms)
- **max_df**: Maximum document frequency (removes common terms)
- **max_k**: Maximum number of clusters to test
- **random_state**: Seed for reproducible results

## 📁 File Structure

```
paper-labeler/
├── cluster_papers_generic.py     # Main clustering script
├── clustering_visualization.html # Web visualization interface
├── serve_generic.py             # Development web server
├── configs/                     # Predefined configurations
│   ├── asplos_config.json
│   ├── arxiv_config.json
│   ├── ieee_config.json
│   ├── pubmed_config.json
│   └── config_template.json
├── results/                     # Generated outputs
│   ├── *_data.json             # JSON for web visualization
│   ├── *_results.csv           # CSV with cluster assignments
│   └── *_plots.png             # Visualization plots
└── README.md                   # This file
```

## 🔧 Command Line Options

### Clustering Script (`cluster_papers_generic.py`)
```bash
positional arguments:
  input_csv             Path to input CSV file containing papers

optional arguments:
  --output-dir DIR      Output directory for results (default: .)
  --output-prefix NAME  Prefix for output files (default: clustering)
  
  # Column mapping
  --title-col COL       Name of title column in CSV
  --author-col COL      Name of author column in CSV
  --summary-col COL     Name of summary/abstract column in CSV
  --keywords-col COL    Name of keywords column in CSV
  --key-col COL         Name of unique key/ID column in CSV
  
  # Clustering parameters
  --max-features N      Maximum number of TF-IDF features
  --max-clusters N      Maximum number of clusters to try
  --show-plots          Display matplotlib plots instead of saving
  
  # Configuration
  --config FILE         Path to JSON configuration file
```

### Web Server (`serve_generic.py`)
```bash
optional arguments:
  --port PORT          Port to serve on (default: 8000)
  --dir DIR            Directory to serve from (default: current)
  --no-browser         Do not open browser automatically
  --html FILE          HTML file to open (default: clustering_visualization.html)
```

## 🌐 Web Interface Features

### Data Upload
- **File Upload**: Upload JSON clustering results directly
- **Format Validation**: Automatic validation of data structure
- **Error Handling**: Clear error messages for invalid data

### Visualization Controls
- **Cluster Filtering**: View specific research areas
- **Search Functionality**: Find papers by title, author, keywords, or content
- **View Switching**: Toggle between scatter plot and network graph
- **Statistics Panel**: Real-time overview of dataset metrics

### Interactive Elements
- **Hover Tooltips**: Detailed paper information on mouse hover
- **Click Actions**: Open paper details in modal/popup
- **Drag & Drop**: Rearrange network nodes
- **Zoom & Pan**: Navigate large visualizations
- **Responsive Layout**: Adapts to different screen sizes

### Network Graph Features
- **Force Simulation**: Physics-based node positioning
- **Cluster Connections**: Links between papers in same cluster
- **Color Coding**: Visual distinction between research areas
- **Interactive Legend**: Cluster overview with paper counts

## 📊 Output Files

### JSON Data (`*_data.json`)
Complete dataset with clustering results for web visualization:
```json
{
  "papers": [...],           // Papers with cluster assignments and coordinates
  "cluster_info": {...},     // Cluster statistics and keywords
  "total_papers": 72,
  "total_clusters": 9,
  "metadata": {
    "source_file": "papers.csv",
    "clustering_method": "K-means",
    "feature_extraction": "TF-IDF",
    "generated_at": "2025-01-01T12:00:00"
  }
}
```

### CSV Results (`*_results.csv`)
Original data with added cluster assignments and PCA coordinates for further analysis.

### Visualization Plots (`*_plots.png`)
High-resolution matplotlib plots showing:
- Cluster scatter plot with PCA projection
- Cluster size distribution
- Explained variance ratios

## 🎨 Customization

### Color Schemes
Edit the `clusterColors` array in `clustering_visualization.html` to customize cluster colors:
```javascript
const clusterColors = [
  '#1f77b4', '#ff7f0e', '#2ca02c', // Your custom colors
  // ... add more colors as needed
];
```

### Clustering Algorithms
The system currently uses K-means clustering, but can be extended to support:
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture Models
- Custom similarity metrics

### Text Processing
Customize text preprocessing by modifying TF-IDF parameters:
- Add domain-specific stop words
- Adjust n-gram ranges for better feature extraction
- Use custom tokenizers for specialized domains

## 🔍 Troubleshooting

### Common Issues

**"Column not found" error**:
- Check CSV column names match configuration
- Use `--title-col`, `--author-col` etc. to map columns
- Examine your CSV file headers

**"No clustering data found"**:
- Ensure JSON file is properly formatted
- Check that required fields (papers, cluster_info) exist
- Validate JSON syntax

**Empty clusters or poor quality**:
- Adjust TF-IDF parameters (min_df, max_df)
- Increase max_features for more detailed analysis
- Check that text fields contain meaningful content

**Web interface not loading**:
- Ensure server is running and accessible
- Check browser console for JavaScript errors
- Verify JSON data is properly formatted

### Data Quality Requirements
- **Minimum Papers**: At least 10-20 papers for meaningful clustering
- **Text Content**: Papers should have substantial text in title/abstract/keywords
- **Language**: Currently optimized for English text
- **Encoding**: CSV files should be UTF-8 encoded

## 🚀 Performance Tips

### For Large Datasets (>1000 papers)
- Increase `max_features` to 2000-5000
- Use `min_df=3` or higher to filter rare terms
- Consider reducing `max_k` to reasonable range (10-30)
- Use `--no-browser` and save plots instead of displaying

### For Domain-Specific Analysis
- Create custom stop words list for your domain
- Adjust n-gram range based on terminology complexity
- Tune clustering parameters based on expected number of topics

## 📚 Examples

### Analyze Computer Science Papers
```bash
# Download ArXiv CS papers and analyze
uv run cluster_papers_generic.py cs_papers.csv \
  --config configs/arxiv_config.json \
  --output-prefix cs_analysis \
  --max-clusters 20
```

### Analyze Medical Literature
```bash
# Process PubMed papers
uv run cluster_papers_generic.py medical_papers.csv \
  --config configs/pubmed_config.json \
  --output-dir medical_results \
  --show-plots
```

### Custom Business Papers
```bash
# Analyze internal company research with custom mapping
uv run cluster_papers_generic.py company_research.csv \
  --title-col "research_title" \
  --summary-col "executive_summary" \
  --keywords-col "tags" \
  --author-col "researchers" \
  --output-prefix company_clusters
```

## 🤝 Contributing

The system is designed to be extensible. Areas for contribution:
- Additional clustering algorithms
- New visualization types
- Support for other document formats
- Improved text preprocessing
- Performance optimizations
- UI/UX enhancements

## 📄 License

Open source - feel free to use, modify, and distribute for academic and commercial purposes.

---

**Made for researchers, by researchers** 🎓