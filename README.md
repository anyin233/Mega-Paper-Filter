# Paper Labeler & Clustering System

> Tired of hunting for the right papers in a sea of conference proceedings? Try this.

A comprehensive system for processing, analyzing, and visualizing academic papers using AI-powered clustering and an interactive web interface.

## Features

- **AI-powered analysis**: Uses the OpenAI API to analyze abstracts and generate summaries and tags.
- **Database management**: SQLite for efficient storage, indexing, and retrieval.
- **Smart clustering**: Automatic clustering with TF–IDF vectorization and K-means (configurable algorithms and parameters).
- **Interactive visualization**: Web interface with scatter plots, network graphs, interactive filtering, and detailed paper views.
- **Multiple input sources**: Import from Zotero CSV, DBLP API, or manual CSV files.
- **Flexible Configuration**: Configurable parameters for different research domains
- **Modern Web Interface**: React frontend with FastAPI backend for real-time updates

## Quick Start (Recommend)

### Step 1: Start the development server

Run the provided startup script:
```bash
./start_dev.sh
```
The script will start the backend and frontend. Note the URL printed to the console (commonly http://localhost:8000).

### Step 2: Open the web UI and import your data

1. Open the URL shown by the script in your browser.
2. Import your papers using the web UI:
  - Use the Import button or drag & drop your file onto the page.
  - Supported formats:
    - Zotero CSV — must include an "abstract" column
    - BibTeX (.bib) — must include an "abstract" field

Important: abstracts are required for AI analysis, tagging, and reliable clustering. If abstracts are missing the system will still import entries but analysis and cluster quality will be reduced.

### Step 3: Review results

After import the app will analyze and cluster your papers. Use the interactive visualizations (scatter/network views, filters, and paper detail panels) to explore and refine the results.

## Run in CLI

### Step 1: Setup
```bash
# Install dependencies
uv sync

# Set up your OpenAI API key (optional, for AI analysis)
echo '{"openai_api_key": "your-key-here", "openai_base_url": "https://api.openai.com/v1"}' > config.json
```

### Step 2: Process Papers
Choose your input method:

**From Zotero CSV:**
```bash
uv run process_zotero_csv.py papers.csv --dataset "My Papers" --force
```

**From DBLP (Conference Search):**
```bash
uv run spider.py
```

**From Database (with AI Analysis):**
```bash
uv run parse_paper_db.py --database papers.db --dataset "My Papers"
```

### Step 3: Generate Clustering
```bash
uv run cluster_from_db.py --database papers.db --dataset "My Papers" --output-prefix analysis
```

### Step 4: Visualize
```bash
# Start web server with auto-detection of JSON files
uv run serve_generic.py

# Or specify a specific JSON file
uv run serve_generic.py analysis_data.json

# Or open static HTML and upload JSON manually
open clustering_visualization.html
```

**Features of the Web Interface:**
- 📊 **Dual Visualization**: Switch between scatter plot and network graph views
- 🔍 **Interactive Filtering**: Search papers and filter by clusters  
- 📄 **Paper Details**: Click any point to see full paper information
- 📤 **File Upload**: Drag & drop JSON files directly into the browser
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Workflows

### Recommended Workflow (Database-Driven)
```
Zotero CSV → Database → AI Analysis → Clustering → Visualization
```

### Alternative Workflow (Direct CSV)
```
CSV File → Direct Clustering → Visualization
```

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for detailed step-by-step instructions.

## Project Structure

```
paper-labeler/
├── backend/                    # FastAPI backend server
│   ├── main.py                # Main server application
│   └── settings.py            # Configuration settings
├── frontend/                  # React frontend application
│   ├── src/                   # Source code
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   └── hooks/             # Custom React hooks
│   └── public/                # Static assets
├── src/                       # Core modules
│   ├── openai_api.py          # OpenAI API integration
│   ├── dblp_api.py            # DBLP database API
│   ├── abstract_extractor.py  # Web scraping utilities
│   └── google_scholar_extractor.py # Google Scholar search
├── cluster_from_db.py         # Database-driven clustering (recommended)
├── cluster_papers_generic.py  # Direct CSV clustering (alternative)
├── process_zotero_csv.py      # Zotero CSV importer
├── parse_paper_db.py          # AI-powered paper analysis
├── db_manager.py              # Database management CLI
├── spider.py                  # Paper collection from DBLP
├── serve_generic.py           # Dynamic web server
└── serve_visualization.py     # Static web server
```

## Key Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `process_zotero_csv.py` | Import Zotero data | CSV file | SQLite database |
| `parse_paper_db.py` | AI analysis of papers | Database | Enhanced database |
| `cluster_from_db.py` | Generate clustering | Database | JSON + plots |
| `spider.py` | Collect papers from DBLP | Interactive | Text file |
| `db_manager.py` | Manage database | Commands | Database operations |

## Documentation

- [QUICK_START.md](QUICK_START.md) - 3-step quick start guide
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - Complete workflow documentation  
- [README_generalized.md](README_generalized.md) - Generic CSV clustering
- [README_clustering.md](README_clustering.md) - Clustering algorithm details
- [ZOTERO_README.md](ZOTERO_README.md) - Zotero integration guide
- [README_frontend.md](README_frontend.md) - Frontend/backend architecture details

## Requirements

- Python 3.12+
- Node.js 16+ (for frontend development)
- OpenAI API key (for AI analysis, optional)
- Modern web browser (for visualization)

## Use Cases

- **Conference Paper Analysis**: Cluster and analyze papers from specific conferences
- **Literature Reviews**: Organize large collections of research papers
- **Research Trend Analysis**: Identify themes and patterns in academic literature
- **Paper Discovery**: Find related papers through clustering visualization

## Contributing

This project uses `uv` for dependency management. Run tests with:
```bash
uv run python -m pytest test/
```

## License

Open source - see repository for details.
