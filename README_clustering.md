# ASPLOS Papers Clustering Visualization

This project provides a comprehensive system for clustering academic papers from the ASPLOS conference and visualizing the results in an interactive web interface.

## Features

### Clustering Analysis
- **Text Processing**: Combines paper titles, summaries, and keywords using TF-IDF vectorization
- **Automatic Clustering**: Uses K-means clustering with optimal cluster number detection via silhouette analysis
- **Dimensionality Reduction**: PCA for 2D visualization
- **Statistical Analysis**: Comprehensive cluster analysis with top keywords and paper samples

### Interactive Web Visualization
- **Dual View Modes**: 
  - Scatter plot showing papers in 2D PCA space
  - Interactive network graph with force-directed layout
- **Filtering & Search**: 
  - Filter by specific clusters
  - Search across titles, authors, and keywords
- **Interactive Elements**:
  - Clickable paper cards with detailed information
  - Hover tooltips with paper details
  - Expandable cluster summaries
  - Real-time statistics

## Files Structure

```
paper-labeler/
├── cluster_papers.py              # Main clustering script
├── clustering_visualization.html  # Web visualization interface
├── clustering_data.json          # Generated clustering data
├── clustered_papers.csv         # CSV output with cluster assignments
├── serve_visualization.py        # Local web server for testing
└── ASPLOS_labeled.csv            # Input dataset
```

## Usage

### 1. Run Clustering Analysis
```bash
uv run cluster_papers.py
```

This will:
- Process the ASPLOS_labeled.csv dataset
- Perform clustering analysis
- Generate visualization plots
- Create clustering_data.json for web interface
- Save clustered_papers.csv with results

### 2. View Web Visualization
```bash
uv run serve_visualization.py
```

This will:
- Start a local web server on port 8000
- Automatically open the visualization in your browser
- Serve the interactive clustering interface

Alternatively, you can manually open `clustering_visualization.html` in any web browser after starting the server.

## Web Interface Features

### Controls
- **Cluster Filter**: View papers from specific clusters only
- **Search**: Find papers by title, author, or keywords
- **View Mode**: Switch between scatter plot and network graph
- **Statistics**: Real-time count of papers and clusters

### Visualizations

#### Scatter Plot View
- Each point represents a paper
- Colors indicate cluster membership
- Interactive hover tooltips
- Based on PCA dimensionality reduction

#### Network Graph View
- Nodes represent papers
- Links connect papers in the same cluster  
- Draggable nodes with force simulation
- Zoom and pan capabilities
- Hover highlighting of connected papers

### Cluster Analysis Panel
- Expandable cluster summaries
- Top keywords for each cluster
- Paper lists with titles, authors, and summaries
- Click-through to paper details

## Clustering Results

The system identifies 9 distinct research clusters in the ASPLOS dataset:

1. **Cluster 0**: Zero-knowledge proofs & GPU acceleration
2. **Cluster 1**: Cloud management & encryption  
3. **Cluster 2**: ML training & GPU communication
4. **Cluster 3**: Mobile devices & rendering
5. **Cluster 4**: Verification & quantum computing
6. **Cluster 5**: Memory systems & storage
7. **Cluster 6**: Analysis & tracing tools
8. **Cluster 7**: Data compression techniques
9. **Cluster 8**: Security & RowHammer attacks

## Technical Details

### Dependencies
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- matplotlib/seaborn: Visualization
- numpy: Numerical computing

### Algorithms Used
- **TF-IDF**: Term frequency-inverse document frequency for text vectorization
- **K-means**: Clustering algorithm with k=9 (automatically determined)
- **PCA**: Principal component analysis for dimensionality reduction
- **Silhouette Analysis**: Optimal cluster number selection

### Web Technologies
- **D3.js**: Interactive network visualization
- **Plotly.js**: Scatter plot visualization
- **Vanilla JavaScript**: Interface interactions
- **CSS Grid/Flexbox**: Responsive layout

## Customization

### Modify Clustering Parameters
Edit `cluster_papers.py` to adjust:
- TF-IDF parameters (max_features, ngram_range)
- Clustering range (max_k in find_optimal_clusters)
- Visualization settings

### Customize Web Interface
Edit `clustering_visualization.html` to modify:
- Color schemes (clusterColors array)
- Layout and styling (CSS)
- Interactive features (JavaScript functions)

## Output Files

- `clustering_data.json`: Complete dataset with cluster assignments and PCA coordinates
- `clustered_papers.csv`: CSV file with original data plus cluster assignments
- Matplotlib plots: Displayed during clustering analysis

## Browser Compatibility

The web interface works with modern browsers supporting:
- ES6 JavaScript features
- CSS Grid and Flexbox
- SVG and Canvas rendering
- WebGL (for advanced visualizations)

## Performance Notes

- Clustering analysis: ~30 seconds for 72 papers
- Web interface: Handles 72 papers smoothly
- Network visualization: Optimized with limited connections to prevent clutter
- Real-time filtering and search across all papers