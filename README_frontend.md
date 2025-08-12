# Paper Labeler - Enhanced Frontend

A modern web application for academic paper management, analysis, and clustering with an interactive frontend.

## Architecture Overview

This enhanced version features:
- **FastAPI Backend**: REST API with WebSocket support for real-time updates
- **React Frontend**: Modern UI with Material-UI components
- **Database Storage**: SQLite database for persistent paper storage
- **Real-time Processing**: WebSocket connections for live progress updates
- **Interactive Visualization**: Plotly-based clustering visualization

## Features

### 1. Paper Management
- **CSV Upload**: Drag-and-drop interface for Zotero exports and other CSV formats
- **Real-time Processing**: Live progress tracking during paper import
- **Search & Filter**: Advanced paper search with dataset filtering
- **Paper Details**: Comprehensive paper information display

### 2. Clustering Analysis
- **Configurable Parameters**: Adjustable TF-IDF features, cluster counts, and minimum papers
- **Progress Monitoring**: Real-time clustering progress with WebSocket updates
- **Multiple Datasets**: Cluster specific datasets or all papers

### 3. Interactive Visualization
- **Scatter Plot**: PCA-projected 2D visualization of paper clusters
- **Interactive Points**: Click papers to view details
- **Cluster Filtering**: View specific clusters or search within results
- **Cluster Analysis**: Detailed cluster statistics and top keywords

### 4. Dashboard
- **Statistics Overview**: Paper counts, dataset information, processing status
- **Charts**: Bar charts and pie charts for dataset distribution
- **Recent Activity**: Latest datasets and processing jobs

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Install Dependencies**:
   ```bash
   # From project root
   uv sync
   ```

2. **Start the Backend Server**:
   ```bash
   uv run python backend/main.py
   ```
   
   The API will be available at `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/api/health`

### Frontend Setup

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the Development Server**:
   ```bash
   npm start
   ```
   
   The frontend will be available at `http://localhost:3000`

### Production Build

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve via Backend**:
   The FastAPI backend will automatically serve the built frontend from `/`

## Usage Workflow

### 1. Upload Papers
1. Navigate to "Upload Papers"
2. Drag and drop a CSV file (Zotero export format supported)
3. Enter dataset name and description
4. Monitor real-time upload progress
5. View results in Paper List

### 2. Run Clustering
1. Navigate to "Clustering"
2. Configure parameters:
   - Select dataset (or use all papers)
   - Adjust max features (100-5000)
   - Set max clusters (2-30)
   - Set minimum papers threshold
3. Click "Start Clustering"
4. Monitor real-time progress via WebSocket
5. View results when complete

### 3. Explore Results
1. Navigate to "Visualization"
2. Interact with the scatter plot:
   - Click points to view paper details
   - Filter by cluster
   - Search within results
3. Analyze cluster characteristics:
   - View top keywords
   - See sample paper titles
   - Understand cluster sizes

## API Endpoints

### Papers
- `GET /api/papers` - Get papers with filtering
- `POST /api/papers/upload` - Upload CSV file
- `GET /api/datasets` - List datasets
- `POST /api/datasets` - Create dataset

### Clustering
- `POST /api/clustering/run` - Start clustering job
- `GET /api/clustering/status/{job_id}` - Get job status
- `GET /api/clustering/results/{job_id}` - Get clustering results

### Real-time
- `WebSocket /ws/progress` - Real-time job updates

## Configuration

### Backend Configuration
The backend can be configured via environment variables:
- `DB_PATH`: Database file path (default: `papers.db`)
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)

### Frontend Configuration
The frontend connects to the backend via:
- `REACT_APP_API_URL`: Backend URL (default: `http://localhost:8000`)

## Database Schema

The application uses SQLite with two main tables:

### Papers Table
- `id`: Primary key
- `paper_id`: Unique paper identifier
- `title`: Paper title
- `authors`: JSON array of authors
- `abstract`: Paper abstract
- `summary`: AI-generated summary
- `keywords`: JSON array of keywords
- `url`, `doi`: Paper identifiers
- `publication_year`: Publication year
- `venue`: Journal/conference
- `source_dataset`: Dataset name
- `content_hash`: Deduplication hash
- `created_at`, `updated_at`: Timestamps

### Datasets Table
- `id`: Primary key
- `name`: Dataset name
- `description`: Dataset description
- `source_file`: Original file path
- `total_papers`, `processed_papers`: Counters
- `created_at`, `updated_at`: Timestamps

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLite**: Database storage
- **Pandas**: Data processing
- **Scikit-learn**: Clustering algorithms
- **Matplotlib/Seaborn**: Data visualization
- **WebSockets**: Real-time communication

### Frontend
- **React 18**: Frontend framework
- **TypeScript**: Type safety
- **Material-UI**: Component library
- **Plotly.js**: Interactive visualizations
- **Recharts**: Chart components
- **Axios**: HTTP client
- **React Dropzone**: File upload

## Development

### Adding New Features
1. **Backend**: Add endpoints in `backend/main.py`
2. **Frontend**: Create components in `frontend/src/components/`
3. **API Client**: Update `frontend/src/services/api.ts`

### Testing
- **Backend**: Use FastAPI's built-in testing with pytest
- **Frontend**: Use React Testing Library

### Deployment
1. Build frontend: `npm run build`
2. Configure backend for production
3. Use reverse proxy (nginx) for static files
4. Consider containerization with Docker

## Troubleshooting

### Common Issues
1. **CORS Errors**: Ensure backend CORS settings include frontend URL
2. **WebSocket Connection**: Check firewall settings for WebSocket traffic
3. **CSV Upload Errors**: Verify CSV format matches expected schema
4. **Memory Issues**: Large datasets may require increased memory limits

### Performance Optimization
1. **Database**: Add indexes for large datasets
2. **Frontend**: Implement pagination for large paper lists
3. **Clustering**: Adjust feature limits for faster processing
4. **Visualization**: Use data sampling for very large clusters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

[Add your license information here]