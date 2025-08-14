#!/usr/bin/env python3
"""
FastAPI backend for the paper labeling application.
Provides REST API and WebSocket endpoints for paper management, processing, and clustering.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
import numpy as np
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
import os
import sys
import hashlib
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

from database import PaperDatabase
from cluster_from_db import DatabaseClusteringAnalyzer
from process_zotero_csv import process_zotero_csv, clean_author_list, clean_keywords_list
from backend.settings import get_settings_manager

# Output directory configuration
OUTPUT_DIR = Path("output")

def ensure_output_directory():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

def get_output_path(filename: str) -> str:
    """Get a path in the output directory for the given filename."""
    ensure_output_directory()
    return str(OUTPUT_DIR / filename)

app = FastAPI(
    title="Paper Labeler API",
    description="Backend API for academic paper management and clustering",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
job_status = {}  # Store background job statuses
active_connections: List[WebSocket] = []  # WebSocket connections for real-time updates
thread_pool = ThreadPoolExecutor(max_workers=2)  # Thread pool for CPU-intensive tasks

# Database path
DB_PATH = "papers.db"

# Pydantic models for API
class DatasetCreate(BaseModel):
    name: str = Field(..., description="Dataset name")
    description: str = Field("", description="Dataset description")

class PaperResponse(BaseModel):
    id: int
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    summary: str
    keywords: List[str]
    url: str
    doi: str
    publication_year: Optional[int]
    venue: str
    source_dataset: str
    created_at: str

class PaginatedPapersResponse(BaseModel):
    papers: List[PaperResponse]
    total: int
    page: int
    page_size: int

class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str
    total_papers: int
    created_at: str

class ClusteringConfig(BaseModel):
    dataset_name: Optional[str] = Field(None, description="Dataset to cluster (None for all)")
    max_features: int = Field(1000, description="Maximum TF-IDF features (for traditional clustering)")
    max_k: int = Field(25, description="Maximum number of clusters to test")  # Increased from 15 to 25
    min_papers: int = Field(5, description="Minimum papers required for clustering")
    clustering_method: str = Field("traditional", description="Clustering method: 'traditional', 'llm', or 'embedding'")
    # Feature extraction parameters
    feature_extraction_method: str = Field("tfidf", description="Feature extraction method: 'tfidf' or 'sentence_transformer'")
    sentence_transformer_model: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model name")
    # Traditional clustering algorithm parameters
    traditional_algorithm: str = Field("kmeans", description="Traditional clustering algorithm: 'kmeans', 'agglomerative', 'dbscan', 'spectral'")
    # DBSCAN parameters
    dbscan_eps: float = Field(0.5, description="DBSCAN epsilon parameter")
    dbscan_min_samples: int = Field(5, description="DBSCAN minimum samples parameter")
    # Agglomerative parameters  
    agglomerative_linkage: str = Field("ward", description="Agglomerative clustering linkage method")
    # Spectral parameters
    spectral_assign_labels: str = Field("discretize", description="Spectral clustering label assignment method")
    # LLM clustering parameters
    llm_model: Optional[str] = Field("gpt-4o", description="LLM model for semantic clustering")
    custom_model_name: Optional[str] = Field(None, description="Custom model name when llm_model is 'custom'")
    max_papers_llm: int = Field(500, description="Maximum papers for LLM clustering (to manage costs)")
    # Embedding clustering parameters
    embedding_model: Optional[str] = Field("text-embedding-ada-002", description="Embedding model for embedding-based clustering")
    embedding_batch_size: int = Field(50, description="Batch size for embedding generation")
    embedding_clustering_algorithm: str = Field("kmeans", description="Clustering algorithm: 'kmeans', 'dbscan', 'agglomerative'")
    embedding_dbscan_eps: float = Field(0.5, description="Embedding DBSCAN epsilon parameter")
    embedding_dbscan_min_samples: int = Field(5, description="Embedding DBSCAN minimum samples parameter")
    embedding_agglomerative_linkage: str = Field("ward", description="Embedding agglomerative clustering linkage method")

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(None, description="OpenAI base URL")
    openai_model: Optional[str] = Field(None, description="OpenAI model name")
    openai_enabled: Optional[bool] = Field(None, description="Enable OpenAI processing")
    auto_generate_summary: Optional[bool] = Field(None, description="Auto-generate summaries")
    auto_generate_keywords: Optional[bool] = Field(None, description="Auto-generate keywords")
    # New embedding model settings
    embedding_api_key: Optional[str] = Field(None, description="Embedding model API key")
    embedding_base_url: Optional[str] = Field(None, description="Embedding model base URL")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    embedding_enabled: Optional[bool] = Field(None, description="Enable embedding model")

class AIProcessRequest(BaseModel):
    paper_ids: List[str] = Field(..., description="List of paper IDs to process")
    generate_summary: bool = Field(True, description="Generate summaries")
    generate_keywords: bool = Field(True, description="Generate keywords")
    overwrite_existing: bool = Field(False, description="Overwrite existing summaries/keywords")

class DatasetMergeRequest(BaseModel):
    source_dataset: str = Field(..., description="Source dataset name to merge from")
    target_dataset: str = Field(..., description="Target dataset name to merge into")
    delete_source: bool = Field(True, description="Delete source dataset after merge")

class UploadToDatasetRequest(BaseModel):
    dataset_name: str = Field(..., description="Existing dataset name to upload to")
    create_if_missing: bool = Field(False, description="Create dataset if it doesn't exist")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Helper functions
def get_db():
    """Get database instance."""
    return PaperDatabase(DB_PATH)

async def update_job_status(job_id: str, status: str, progress: float = None, 
                          message: str = "", result: Any = None, error: str = None):
    """Update job status and notify WebSocket clients."""
    if job_id not in job_status:
        job_status[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "message": "",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    # Update status
    job_status[job_id]["status"] = status
    job_status[job_id]["updated_at"] = datetime.now().isoformat()
    if progress is not None:
        job_status[job_id]["progress"] = progress
    if message:
        job_status[job_id]["message"] = message
    if result is not None:
        job_status[job_id]["result"] = result
    if error:
        job_status[job_id]["error"] = error
    
    # Broadcast final status to WebSocket clients
    await manager.broadcast(json.dumps({
        "type": "job_update",
        "data": job_status[job_id]
    }))
    
    # Clean up finished jobs after a short delay to allow clients to receive final status
    if status in ["completed", "failed"]:
        logger.info(f"Job {job_id} finished with status '{status}'. Scheduling cleanup in 30 seconds.")
        
        async def cleanup_job():
            await asyncio.sleep(30)  # Give clients time to receive the final status
            if job_id in job_status:
                logger.info(f"Cleaning up finished job {job_id} from memory")
                del job_status[job_id]
        
        # Schedule cleanup as a background task
        asyncio.create_task(cleanup_job())

# API Endpoints

@app.get("/", response_class=FileResponse)
async def read_root():
    """Serve the React frontend."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "build" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {"message": "Paper Labeler API", "docs": "/docs"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Dataset endpoints
@app.get("/api/datasets", response_model=List[DatasetResponse])
async def get_datasets():
    """Get all datasets with statistics."""
    db = get_db()
    try:
        datasets = db.get_datasets()
        return [
            DatasetResponse(
                id=d["id"],
                name=d["name"],
                description=d["description"] or "",
                total_papers=d["actual_papers"],
                created_at=d["created_at"]
            )
            for d in datasets
        ]
    finally:
        db.close()

@app.post("/api/datasets", response_model=DatasetResponse)
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset."""
    db = get_db()
    try:
        dataset_id = db.create_dataset(dataset.name, dataset.description)
        
        # Get the created dataset
        datasets = db.get_datasets()
        created_dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        
        if not created_dataset:
            raise HTTPException(status_code=500, detail="Failed to create dataset")
        
        return DatasetResponse(
            id=created_dataset["id"],
            name=created_dataset["name"],
            description=created_dataset["description"] or "",
            total_papers=created_dataset["actual_papers"],
            created_at=created_dataset["created_at"]
        )
    finally:
        db.close()

@app.post("/api/datasets/merge")
async def merge_datasets(request: DatasetMergeRequest):
    """Merge one dataset into another."""
    db = get_db()
    try:
        # Verify both datasets exist
        datasets = {d["name"]: d for d in db.get_datasets()}
        
        if request.source_dataset not in datasets:
            raise HTTPException(status_code=404, detail=f"Source dataset '{request.source_dataset}' not found")
        
        if request.target_dataset not in datasets:
            raise HTTPException(status_code=404, detail=f"Target dataset '{request.target_dataset}' not found")
        
        if request.source_dataset == request.target_dataset:
            raise HTTPException(status_code=400, detail="Source and target datasets cannot be the same")
        
        # Perform the merge
        stats = db.merge_datasets(
            source_dataset=request.source_dataset,
            target_dataset=request.target_dataset,
            delete_source=request.delete_source
        )
        
        return {
            "message": f"Successfully merged '{request.source_dataset}' into '{request.target_dataset}'",
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")
    finally:
        db.close()

# Paper endpoints
@app.get("/api/papers", response_model=PaginatedPapersResponse)
async def get_papers(
    dataset: Optional[str] = Query(None, description="Filter by dataset name"),
    search: Optional[str] = Query(None, description="Search query"),
    limit: int = Query(100, ge=1, le=1000, description="Number of papers to return"),
    offset: int = Query(0, ge=0, description="Number of papers to skip")
):
    """Get papers with optional filtering and pagination."""
    db = get_db()
    try:
        if search:
            all_papers = db.search_papers(search, dataset)
        elif dataset:
            all_papers = db.get_papers_by_dataset(dataset)
        else:
            all_papers = db.get_all_papers()
        
        total_count = len(all_papers)
        
        # Apply pagination
        papers = all_papers[offset:offset + limit]
        
        return PaginatedPapersResponse(
            papers=[
                PaperResponse(
                    id=p["id"],
                    paper_id=p["paper_id"],
                    title=p["title"],
                    authors=json.loads(p["authors"]) if p["authors"] else [],
                    abstract=p["abstract"] or "",
                    summary=p["summary"] or "",
                    keywords=json.loads(p["keywords"]) if p["keywords"] else [],
                    url=p["url"] or "",
                    doi=p["doi"] or "",
                    publication_year=p["publication_year"],
                    venue=p["venue"] or "",
                    source_dataset=p["source_dataset"],
                    created_at=p["created_at"]
                )
                for p in papers
            ],
            total=total_count,
            page=offset // limit,
            page_size=limit
        )
    finally:
        db.close()

@app.post("/api/papers/upload")
async def upload_papers(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="Dataset name for uploaded papers"),
    description: str = Query("", description="Dataset description"),
    upload_to_existing: bool = Query(False, description="Upload to existing dataset (enable deduplication)")
):
    """Upload a CSV or BibTeX file containing papers."""
    # Validate file type
    file_extension = Path(file.filename).suffix.lower()
    supported_extensions = ['.csv', '.bib', '.bibtex']
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {', '.join(supported_extensions)}"
        )
    
    # If uploading to existing dataset, verify it exists
    if upload_to_existing:
        db = get_db()
        try:
            datasets = {d["name"]: d for d in db.get_datasets()}
            if dataset_name not in datasets:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found. Create it first or disable 'upload_to_existing'.")
        finally:
            db.close()
    
    # Create a job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily with appropriate extension
    suffix = file_extension if file_extension in ['.bib', '.bibtex'] else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        temp_path = tmp_file.name
    
    # Start background processing based on file type
    if file_extension in ['.bib', '.bibtex']:
        background_tasks.add_task(process_bibtex_file_async, job_id, temp_path, dataset_name, description, upload_to_existing)
        message = "BibTeX file upload started"
    else:
        background_tasks.add_task(process_csv_file, job_id, temp_path, dataset_name, description, upload_to_existing)
        message = "CSV file upload started"
    
    return {"job_id": job_id, "message": message, "status": "pending"}

async def process_csv_file(job_id: str, file_path: str, dataset_name: str, description: str, upload_to_existing: bool = False):
    """Background task to process uploaded CSV file."""
    try:
        await update_job_status(job_id, "running", 0.1, "Starting CSV processing...")
        
        db = get_db()
        try:
            if upload_to_existing:
                # Enhanced processing with deduplication for existing datasets
                stats = await process_csv_with_deduplication(
                    file_path, db, dataset_name, description, job_id
                )
            else:
                # Standard processing (creates new dataset)
                stats = process_zotero_csv(
                    file_path, 
                    db, 
                    dataset_name, 
                    description, 
                    dry_run=False
                )
            
            await update_job_status(job_id, "completed", 1.0, "CSV processing completed", result=stats)
            
        finally:
            db.close()
            
    except Exception as e:
        await update_job_status(job_id, "failed", 0.0, f"Error processing CSV: {str(e)}", error=str(e))
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

async def process_csv_with_deduplication(file_path: str, db: PaperDatabase, dataset_name: str, description: str, job_id: str) -> Dict[str, Any]:
    """Process CSV with enhanced deduplication for existing datasets."""
    import pandas as pd
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        for encoding in ['latin-1', 'cp1252', 'utf-16']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                continue
        else:
            raise ValueError("Could not read CSV file with any common encoding")
    
    await update_job_status(job_id, "running", 0.2, f"Processing {len(df)} papers from CSV...")
    
    # Filter for valid paper types
    if 'Item Type' in df.columns:
        valid_types = ['conferencePaper', 'journalArticle']
        df = df[df['Item Type'].isin(valid_types)]
    
    stats = {
        "total_papers": len(df),
        "added_papers": 0,
        "duplicate_papers": 0,
        "error_papers": 0,
        "errors": []
    }
    
    # Process papers with deduplication
    for idx, row in df.iterrows():
        try:
            # Extract paper data
            title = str(row.get('Title', '')).strip()
            if not title:
                stats["error_papers"] += 1
                continue
                
            authors = clean_author_list(str(row.get('Author', '')))
            abstract = str(row.get('Abstract Note', ''))
            url = str(row.get('Url', ''))
            doi = str(row.get('DOI', ''))
            venue = str(row.get('Publication Title', ''))
            
            # Try to parse year
            year = None
            year_str = str(row.get('Date', ''))
            if year_str and year_str.isdigit():
                year = int(year_str)
            
            # Generate paper ID from DOI or title
            paper_id = doi if doi else hashlib.md5(title.encode()).hexdigest()[:8]
            
            # Use enhanced deduplication
            result = db.add_paper_with_deduplication(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                doi=doi,
                publication_year=year,
                venue=venue,
                source_dataset=dataset_name
            )
            
            if result["status"] == "added":
                stats["added_papers"] += 1
            elif result["status"] == "duplicate":
                stats["duplicate_papers"] += 1
            else:
                stats["error_papers"] += 1
                stats["errors"].append(result["message"])
                
        except Exception as e:
            stats["error_papers"] += 1
            stats["errors"].append(f"Row {idx}: {str(e)}")
        
        # Update progress
        progress = 0.2 + (0.7 * (idx + 1) / len(df))
        await update_job_status(
            job_id, "running", progress, 
            f"Processed {idx + 1}/{len(df)} papers. Added: {stats['added_papers']}, Duplicates: {stats['duplicate_papers']}"
        )
    
    # Update dataset statistics
    db.update_dataset_stats(dataset_name)
    
    await update_job_status(job_id, "running", 0.95, "Finalizing...")
    
    return stats

async def process_bibtex_file_async(job_id: str, file_path: str, dataset_name: str, description: str, upload_to_existing: bool = False):
    """Background task to process uploaded BibTeX file."""
    try:
        await update_job_status(job_id, "running", 0.1, "Starting BibTeX processing...")
        
        # Import BibTeX processing functions
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from parse_bibtex import process_bibtex_file
        
        db = get_db()
        try:
            # Process BibTeX file
            result = process_bibtex_file(
                file_path, 
                db, 
                dataset_name, 
                description,
                dry_run=False,
                upload_to_existing=upload_to_existing
            )
            
            await update_job_status(job_id, "completed", 1.0, "BibTeX processing completed", result=result['statistics'])
            
        finally:
            db.close()
            
    except Exception as e:
        await update_job_status(job_id, "failed", 0.0, f"Error processing BibTeX: {str(e)}", error=str(e))
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

# Clustering endpoints
@app.post("/api/clustering/run")
async def run_clustering(background_tasks: BackgroundTasks, config: ClusteringConfig):
    """Start clustering analysis."""
    job_id = str(uuid.uuid4())
    
    # Start background clustering
    background_tasks.add_task(perform_clustering, job_id, config)
    
    return {"job_id": job_id, "message": "Clustering started", "status": "pending"}

async def perform_clustering(job_id: str, config: ClusteringConfig):
    """Background task to perform clustering analysis."""
    try:
        await update_job_status(job_id, "running", 0.1, "Initializing clustering...")
        
        # Check clustering method
        if config.clustering_method == "llm":
            # Perform LLM-based clustering
            await perform_llm_clustering(job_id, config)
        elif config.clustering_method == "embedding":
            # Perform embedding-based clustering
            await perform_embedding_clustering(job_id, config)
        else:
            # Perform traditional clustering
            await perform_traditional_clustering(job_id, config)
            
    except Exception as e:
        await update_job_status(job_id, "failed", 0.0, f"Clustering failed: {str(e)}", error=str(e))

async def perform_traditional_clustering(job_id: str, config: ClusteringConfig):
    """Background task to perform traditional clustering analysis."""
    # Create clustering configuration with enhanced keyword weighting
    clustering_config = {
        'feature_extraction': {
            'method': config.feature_extraction_method,
            'sentence_transformer_model': config.sentence_transformer_model
        },
        'tfidf_params': {
            'max_features': config.max_features,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,  # Minimum documents for a term to be included
            'max_df': 0.7,  # Reduced from 0.8 to give more weight to specific terms like keywords
            'sublinear_tf': True,  # Use log-scaled term frequency
            'norm': 'l2'  # L2 normalization to help with keyword emphasis
        },
        'clustering_params': {
            'method': config.traditional_algorithm,
            'max_k': config.max_k,
            'random_state': 42,
            'n_init': 10,
            # DBSCAN parameters
            'eps': config.dbscan_eps,
            'min_samples': config.dbscan_min_samples,
            # Agglomerative parameters
            'linkage': config.agglomerative_linkage,
            # Spectral parameters
            'assign_labels': config.spectral_assign_labels
        },
        'pca_params': {
            'n_components': 2,
            'random_state': 42
        }
    }
    
    # Initialize analyzer
    analyzer = DatabaseClusteringAnalyzer(DB_PATH, clustering_config)
    
    try:
        await update_job_status(job_id, "running", 0.2, "Loading papers from database...")
        
        # Load papers
        df = analyzer.load_papers_from_db(config.dataset_name, config.min_papers)
        
        await update_job_status(job_id, "running", 0.4, "Creating features...")
        analyzer.create_features()
        
        await update_job_status(job_id, "running", 0.6, "Finding optimal clusters...")
        optimal_k, inertias, silhouette_scores, k_range = analyzer.find_optimal_clusters()
        
        await update_job_status(job_id, "running", 0.7, "Performing clustering...")
        analyzer.perform_clustering(optimal_k)
        
        await update_job_status(job_id, "running", 0.8, "Creating visualizations...")
        analyzer.create_pca_projection()
        
        await update_job_status(job_id, "running", 0.85, "Analyzing clusters...")
        analyzer.analyze_clusters()
        
        # Generate cluster names using LLM if OpenAI is configured
        settings_manager = get_settings_manager()
        openai_config = settings_manager.get_openai_config()
        
        if openai_config.get('enabled') and openai_config.get('api_key'):
            await update_job_status(job_id, "running", 0.9, "Generating cluster names...")
            await analyzer.generate_cluster_names(openai_config)
        else:
            await update_job_status(job_id, "running", 0.9, "Skipping cluster naming (OpenAI not configured)...")
        
        await update_job_status(job_id, "running", 0.95, "Finalizing results...")
        
        # Generate visualization data
        output_path = get_output_path(f"clustering_results_{job_id}.json")
        json_data = analyzer.generate_visualization_data(output_path, config.dataset_name)
        
        # Save clustering result to database
        try:
            db = get_db()
            dataset_name = config.dataset_name or "All Datasets"
            result_name = f"Traditional Clustering - {dataset_name} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
            result_description = f"K-means clustering with {config.max_features} features, max {config.max_k} clusters"
            
            db.save_clustering_result(
                job_id=job_id,
                name=result_name,
                description=result_description,
                dataset_filter=config.dataset_name,
                total_papers=len(df),
                total_clusters=optimal_k,
                clustering_method="K-means",
                feature_extraction="TF-IDF",
                max_features=config.max_features,
                max_k=config.max_k,
                min_papers=config.min_papers,
                silhouette_score=max(silhouette_scores) if silhouette_scores else None,
                pca_explained_variance=analyzer.pca.explained_variance_ratio_.tolist() if hasattr(analyzer, 'pca') else None,
                visualization_data=json_data
            )
            db.close()
        except Exception as e:
            logger.error(f"Failed to save clustering result to database: {e}")
        
        await update_job_status(
            job_id, 
            "completed", 
            1.0, 
            "Traditional clustering completed successfully",
            result={
                "output_file": output_path,
                "total_papers": len(df),
                "total_clusters": optimal_k,
                "silhouette_score": max(silhouette_scores) if silhouette_scores else None,
                "visualization_data": json_data,
                "cluster_names_generated": analyzer.cluster_names is not None,
                "clustering_method": "traditional"
            }
        )
        
    finally:
        analyzer.close()

async def perform_llm_clustering(job_id: str, config: ClusteringConfig):
    """Background task to perform LLM-based semantic clustering."""
    # Get OpenAI configuration
    settings_manager = get_settings_manager()
    openai_config = settings_manager.get_openai_config()
    
    if not openai_config.get('enabled') or not openai_config.get('api_key'):
        raise ValueError("OpenAI is not configured or enabled. LLM clustering requires OpenAI API access.")
    
    # Override model if specified in config
    if config.llm_model == 'custom':
        if not config.custom_model_name or not config.custom_model_name.strip():
            raise ValueError("Custom model name is required when 'custom' model is selected.")
        openai_config['model'] = config.custom_model_name.strip()
    elif config.llm_model:
        openai_config['model'] = config.llm_model
    
    # Import LLM clustering analyzer
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from llm_cluster_papers import LLMClusteringAnalyzer
    
    # Initialize LLM analyzer
    analyzer = LLMClusteringAnalyzer(DB_PATH, openai_config)
    
    try:
        await update_job_status(job_id, "running", 0.2, "Loading papers for LLM clustering...")
        
        # Load papers (limited for LLM processing)
        df = analyzer.load_papers_from_db(
            dataset_name=config.dataset_name, 
            min_papers=config.min_papers,
            max_papers=config.max_papers_llm
        )
        
        await update_job_status(job_id, "running", 0.4, "Performing LLM-based semantic clustering...")
        
        # Perform LLM clustering
        clustering_result = await analyzer.perform_llm_clustering(config.max_k)
        
        await update_job_status(job_id, "running", 0.8, "Generating visualization data...")
        
        # Generate visualization data
        output_path = get_output_path(f"llm_clustering_results_{job_id}.json")
        json_data = analyzer.generate_visualization_data(output_path, config.dataset_name)
        
        await update_job_status(job_id, "running", 0.9, "Saving results to database...")
        
        # Save clustering result to database
        try:
            db = get_db()
            dataset_name = config.dataset_name or "All Datasets"
            result_name = f"LLM Clustering - {dataset_name} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
            result_description = f"LLM semantic clustering with {openai_config.get('model', 'gpt-4o')} model"
            
            db.save_clustering_result(
                job_id=job_id,
                name=result_name,
                description=result_description,
                dataset_filter=config.dataset_name,
                total_papers=len(df),
                total_clusters=clustering_result['total_clusters'],
                clustering_method=f"LLM-{openai_config.get('model', 'gpt-4o')}",
                feature_extraction="OpenAI Language Model",
                max_features=None,  # Not applicable for LLM
                max_k=config.max_k,
                min_papers=config.min_papers,
                silhouette_score=None,  # Not applicable for LLM clustering
                pca_explained_variance=None,  # Not applicable for LLM clustering
                visualization_data=json_data
            )
            db.close()
        except Exception as e:
            logger.error(f"Failed to save LLM clustering result to database: {e}")
        
        await update_job_status(
            job_id, 
            "completed", 
            1.0, 
            "LLM clustering completed successfully",
            result={
                "output_file": output_path,
                "total_papers": len(df),
                "total_clusters": clustering_result['total_clusters'],
                "visualization_data": json_data,
                "clustering_method": "llm",
                "llm_model": openai_config.get('model', 'gpt-4o'),
                "assigned_papers": clustering_result.get('assigned_papers', 0),
                "unassigned_papers": clustering_result.get('unassigned_papers', 0)
            }
        )
        
    finally:
        analyzer.close()

async def perform_embedding_clustering(job_id: str, config: ClusteringConfig):
    """Background task to perform embedding-based clustering analysis."""
    # Import embedding clustering analyzer
    from embedding_clustering import EmbeddingClusteringAnalyzer
    
    # Get settings
    settings_manager = get_settings_manager()
    openai_config = settings_manager.get_openai_config()
    embedding_config = settings_manager.get_embedding_config()
    
    if not embedding_config["enabled"]:
        raise ValueError("Embedding model not configured or disabled")
    
    await update_job_status(job_id, "running", 0.2, "Setting up embedding clustering...")
    
    # Create embedding-specific clustering configuration
    clustering_config = {
        'embedding_params': {
            'model': config.embedding_model or embedding_config['model'],
            'batch_size': config.embedding_batch_size,
            'max_concurrent': 3,
            'cache_embeddings': True
        },
        'clustering_params': {
            'max_k': config.max_k,
            'random_state': 42,
            'method': config.embedding_clustering_algorithm,
            'dbscan_eps': config.dbscan_eps,
            'dbscan_min_samples': config.dbscan_min_samples,
            'agglomerative_linkage': config.agglomerative_linkage
        },
        'pca_params': {
            'n_components': 2,
            'random_state': 42
        },
        'dataset_name': config.dataset_name
    }
    
    def run_clustering_sync():
        """Synchronous clustering function to run in thread pool."""
        # Create a new database connection within the thread
        thread_db = PaperDatabase(DB_PATH)
        analyzer = None
        
        try:
            # Initialize analyzer with thread-local database connection
            analyzer = EmbeddingClusteringAnalyzer(DB_PATH, clustering_config)
            
            # Load papers from database
            df = analyzer.load_papers_from_db(config.dataset_name, config.min_papers)
            
            # Setup embedding client
            analyzer.setup_embedding_client(embedding_config)
            
            # Create embeddings
            analyzer.create_embeddings()
            
            # Find optimal clusters and perform clustering
            optimal_k, inertias, silhouette_scores, k_range = analyzer.find_optimal_clusters()
            clusterer, cluster_labels = analyzer.perform_clustering(optimal_k)
            
            # Create PCA projection
            analyzer.create_pca_projection()
            
            # Analyze clusters
            analyzer.analyze_clusters()
            
            # Generate visualization data
            output_path = get_output_path(f"clustering_embedding_{job_id}.json")
            visualization_data = analyzer.generate_visualization_data(output_path, config.dataset_name)
            
            # Save clustering results to database using thread-local connection
            try:
                result_name = f"Embedding Clustering - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                result_description = f"Embedding-based clustering using {embedding_config['model']} with {config.embedding_clustering_algorithm.upper()}"
                
                clustering_result_id = thread_db.save_clustering_result(
                    job_id=job_id,
                    name=result_name,
                    description=result_description,
                    dataset_filter=config.dataset_name,
                    total_papers=len(analyzer.df),
                    total_clusters=len(np.unique(cluster_labels)),
                    clustering_method=f"Embedding + {config.embedding_clustering_algorithm.upper()}",
                    feature_extraction=f"LLM Embeddings ({embedding_config['model']})",
                    silhouette_score=max(silhouette_scores) if silhouette_scores else 0.0,
                    visualization_data=visualization_data
                )
                
                logger.info(f"Saved embedding clustering result to database with ID: {clustering_result_id}")
            except Exception as e:
                logger.error(f"Failed to save embedding clustering result to database: {e}")
                clustering_result_id = None
            
            return analyzer, visualization_data, cluster_labels, silhouette_scores, optimal_k, clustering_result_id
            
        except Exception as e:
            if analyzer:
                analyzer.close()
            raise e
        finally:
            # Always close the thread-local database connection
            thread_db.close()
    
    try:
        # Update progress before starting CPU-intensive work
        await update_job_status(job_id, "running", 0.3, "Loading papers and setting up embedding client...")
        
        # Run the synchronous clustering operation in thread pool
        loop = asyncio.get_event_loop()
        analyzer, visualization_data, cluster_labels, silhouette_scores, optimal_k, clustering_result_id = await loop.run_in_executor(
            thread_pool, run_clustering_sync
        )
        
        # Update progress after clustering is complete
        await update_job_status(job_id, "running", 0.9, "Generating cluster names...")
        
        # Generate cluster names using LLM if available (this is async, so run normally)
        if openai_config["enabled"]:
            await analyzer.generate_cluster_names(openai_config)
        
        await update_job_status(job_id, "running", 0.95, "Finalizing results...")
        
        # Complete the job
        await update_job_status(
            job_id, "completed", 1.0, 
            "Embedding clustering analysis completed successfully!",
            result={
                "visualization_data": visualization_data,
                "clustering_result_id": clustering_result_id,
                "method": f"Embedding + {config.embedding_clustering_algorithm.upper()}",
                "total_papers": len(analyzer.df),
                "total_clusters": len(np.unique(cluster_labels)),
                "silhouette_score": max(silhouette_scores) if silhouette_scores else 0.0,
                "embedding_model": embedding_config['model'],
                "clustering_algorithm": config.embedding_clustering_algorithm
            }
        )
        
    except Exception as e:
        logger.error(f"Embedding clustering failed: {e}")
        await update_job_status(job_id, "failed", 0.0, f"Embedding clustering failed: {str(e)}", error=str(e))
    finally:
        if 'analyzer' in locals():
            analyzer.close()

@app.get("/api/clustering/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a clustering job."""
    if job_id not in job_status:
        # Job might have been cleaned up because it finished
        raise HTTPException(status_code=404, detail="Job not found or has been cleaned up (job completed)")
    
    return JobStatus(**job_status[job_id])

@app.get("/api/clustering/active-jobs")
async def get_active_jobs():
    """Get all active and recent clustering jobs."""
    # Return jobs from last 24 hours, prioritizing active ones
    from datetime import datetime, timedelta
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    recent_jobs = []
    for job_id, status in job_status.items():
        try:
            job_updated = datetime.fromisoformat(status["updated_at"])
            if job_updated > cutoff_time:
                recent_jobs.append({
                    "job_id": job_id,
                    "status": status["status"],
                    "progress": status["progress"],
                    "message": status["message"],
                    "created_at": status["created_at"],
                    "updated_at": status["updated_at"],
                    "has_result": status["result"] is not None,
                    "has_error": status["error"] is not None
                })
        except (ValueError, KeyError):
            continue
    
    # Sort by updated time, most recent first
    recent_jobs.sort(key=lambda x: x["updated_at"], reverse=True)
    
    return {
        "jobs": recent_jobs,
        "active_count": len([j for j in recent_jobs if j["status"] in ["pending", "running"]]),
        "total_count": len(recent_jobs)
    }

@app.get("/api/clustering/results/{job_id}")
async def get_clustering_results(job_id: str):
    """Get clustering results."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    if not job["result"]:
        raise HTTPException(status_code=404, detail="No results found")
    
    return job["result"]

@app.get("/api/clustering/saved-results")
async def get_saved_clustering_results():
    """Get all saved clustering results."""
    db = get_db()
    try:
        results = db.get_clustering_results()
        return results
    finally:
        db.close()

@app.get("/api/clustering/saved-results/{result_id}")
async def get_saved_clustering_result(result_id: int):
    """Get a specific saved clustering result."""
    db = get_db()
    try:
        result = db.get_clustering_result(result_id=result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Clustering result not found")
        return result
    finally:
        db.close()

@app.delete("/api/clustering/saved-results/{result_id}")
async def delete_saved_clustering_result(result_id: int):
    """Delete a saved clustering result."""
    db = get_db()
    try:
        success = db.delete_clustering_result(result_id)
        if not success:
            raise HTTPException(status_code=404, detail="Clustering result not found")
        return {"message": "Clustering result deleted successfully"}
    finally:
        db.close()

@app.get("/api/clustering/saved-results/by-dataset/{dataset_name}")
async def get_clustering_results_by_dataset(dataset_name: str):
    """Get clustering results filtered by dataset."""
    db = get_db()
    try:
        results = db.get_clustering_results_by_dataset(dataset_name)
        return results
    finally:
        db.close()

# Statistics endpoint
@app.get("/api/statistics")
async def get_statistics():
    """Get database statistics."""
    db = get_db()
    try:
        stats = db.get_statistics()
        return stats
    finally:
        db.close()

# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    """Get application settings (with sensitive data masked)."""
    settings_manager = get_settings_manager()
    return settings_manager.get_all_settings()

@app.post("/api/settings")
async def update_settings(settings_update: SettingsUpdate):
    """Update application settings."""
    settings_manager = get_settings_manager()
    
    # Update OpenAI settings
    if any([settings_update.openai_api_key is not None,
            settings_update.openai_base_url is not None,
            settings_update.openai_model is not None,
            settings_update.openai_enabled is not None]):
        
        settings_manager.set_openai_config(
            api_key=settings_update.openai_api_key,
            base_url=settings_update.openai_base_url,
            model=settings_update.openai_model,
            enabled=settings_update.openai_enabled
        )
    
    # Update processing settings
    if settings_update.auto_generate_summary is not None:
        settings_manager.set_setting("processing.auto_generate_summary", 
                                   settings_update.auto_generate_summary)
    
    if settings_update.auto_generate_keywords is not None:
        settings_manager.set_setting("processing.auto_generate_keywords", 
                                   settings_update.auto_generate_keywords)
    
    # Update embedding settings
    if any([settings_update.embedding_api_key is not None,
            settings_update.embedding_base_url is not None,
            settings_update.embedding_model is not None,
            settings_update.embedding_enabled is not None]):
        
        settings_manager.set_embedding_config(
            api_key=settings_update.embedding_api_key,
            base_url=settings_update.embedding_base_url,
            model=settings_update.embedding_model,
            enabled=settings_update.embedding_enabled
        )
    
    return {"message": "Settings updated successfully"}

@app.post("/api/settings/test-openai")
async def test_openai_connection():
    """Test OpenAI API connection."""
    settings_manager = get_settings_manager()
    result = settings_manager.test_openai_connection()
    
    if not result["success"]:
        logger.error(f"OpenAI connection test failed: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/api/settings/test-embedding")
async def test_embedding_connection():
    """Test embedding model API connection."""
    settings_manager = get_settings_manager()
    result = settings_manager.test_embedding_connection()
    
    if not result["success"]:
        logger.error(f"Embedding connection test failed: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

# AI Processing endpoints
@app.post("/api/ai/process")
async def process_papers_ai(request: AIProcessRequest):
    """Process papers with AI to generate summaries and keywords."""
    settings_manager = get_settings_manager()
    openai_config = settings_manager.get_openai_config()
    
    if not openai_config["enabled"]:
        raise HTTPException(status_code=400, detail="OpenAI is not configured or enabled")
    
    # Add concurrent processing settings to config
    processing_settings = settings_manager.get_setting("processing", {})
    openai_config["max_concurrent"] = processing_settings.get("concurrent_requests", 3)
    
    job_id = str(uuid.uuid4())
    
    # Start background AI processing using asyncio.create_task for true async
    asyncio.create_task(
        perform_ai_processing(
            job_id, 
            request.paper_ids,
            request.generate_summary,
            request.generate_keywords,
            request.overwrite_existing,
            openai_config
        )
    )
    
    return {"job_id": job_id, "message": "AI processing started", "status": "pending"}

async def perform_ai_processing(job_id: str, paper_ids: List[str], 
                              generate_summary: bool, generate_keywords: bool,
                              overwrite_existing: bool, openai_config: Dict[str, Any]):
    """Background task to perform AI processing on papers."""
    try:
        await update_job_status(job_id, "running", 0.0, "Initializing AI processing...")
        
        # Import OpenAI functions
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from src.openai_api import get_async_openai_client, get_async_openai_response
        
        # Create async client
        client = get_async_openai_client(openai_config["api_key"], openai_config["base_url"])
        
        # Create semaphore for concurrency control (limit concurrent requests)
        max_concurrent = openai_config.get("max_concurrent", 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        db = get_db()
        processed_count = 0
        total_papers = len(paper_ids)
        results = {"processed": 0, "skipped": 0, "errors": 0, "error_details": []}
        
        async def process_single_paper(i: int, paper_id: str) -> Dict[str, Any]:
            """Process a single paper with semaphore control."""
            async with semaphore:  # Limit concurrent API calls
                try:
                    # Get paper from database
                    paper = db.get_paper(paper_id)
                    if not paper:
                        return {"status": "skipped", "reason": "Paper not found"}
                    
                    # Check if processing is needed
                    needs_summary = generate_summary and (
                        overwrite_existing or not paper.get("summary") or paper.get("summary").strip() == ""
                    )
                    needs_keywords = generate_keywords and (
                        overwrite_existing or not paper.get("keywords") or 
                        json.loads(paper.get("keywords", "[]")) == []
                    )
                    
                    if not needs_summary and not needs_keywords:
                        return {"status": "skipped", "reason": "No processing needed"}
                    
                    # Create prompt for AI
                    title = paper.get("title", "")
                    abstract = paper.get("abstract", "")
                    
                    if not title and not abstract:
                        return {"status": "skipped", "reason": "No title or abstract"}
                    
                    prompt = f"Title: {title}\n\nAbstract: {abstract}"
                    
                    await update_job_status(
                        job_id, "running", 
                        (i + 0.5) / total_papers, 
                        f"Processing paper {i + 1}/{total_papers}: {title[:50]}..."
                    )
                    
                    # Get AI response (async)
                    response_text = await get_async_openai_response(client, prompt, openai_config["model"])
                    
                    # Use robust JSON parsing instead of direct json.loads
                    from src.openai_api import safe_json_parse
                    response_data = safe_json_parse(response_text, fallback_dict={
                        "summary": f"AI processing failed for: {title[:100]}...",
                        "keywords": ["processing-error"]
                    })
                    
                    # Extract summary and keywords
                    new_summary = response_data.get("summary", "") if needs_summary else paper.get("summary", "")
                    new_keywords = response_data.get("keywords", []) if needs_keywords else json.loads(paper.get("keywords", "[]"))
                    
                    # Update paper in database
                    db.update_paper_summary(paper_id, new_summary, new_keywords)
                    
                    return {"status": "processed", "paper_id": paper_id}
                    
                except Exception as e:
                    error_msg = f"Paper {paper_id}: {str(e)}"
                    return {"status": "error", "error": error_msg, "paper_id": paper_id}
        
        # Process papers with concurrency control
        tasks = [process_single_paper(i, paper_id) for i, paper_id in enumerate(paper_ids)]
        
        # Process papers in batches to avoid overwhelming the API
        batch_size = max_concurrent * 2
        processed_count = 0
        
        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            # Process current batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                global_i = batch_start + i
                
                if isinstance(result, Exception):
                    results["errors"] += 1
                    results["error_details"].append(f"Paper {paper_ids[global_i]}: {str(result)}")
                elif result["status"] == "processed":
                    results["processed"] += 1
                    processed_count += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                elif result["status"] == "error":
                    results["errors"] += 1
                    results["error_details"].append(result["error"])
                
                # Update progress after each paper
                await update_job_status(
                    job_id, "running", 
                    (global_i + 1) / total_papers, 
                    f"Processed {processed_count}/{total_papers} papers (batch {batch_start // batch_size + 1})"
                )
        
        # Final status update
        await update_job_status(
            job_id, "completed", 1.0, 
            f"AI processing completed. Processed: {results['processed']}, Skipped: {results['skipped']}, Errors: {results['errors']}",
            result=results
        )
        
    except Exception as e:
        await update_job_status(job_id, "failed", 0.0, f"AI processing failed: {str(e)}", error=str(e))

@app.get("/api/papers/unprocessed")
async def get_unprocessed_papers(
    dataset: Optional[str] = Query(None, description="Filter by dataset name"),
    missing_summary: bool = Query(True, description="Include papers without summary"),
    missing_keywords: bool = Query(True, description="Include papers without keywords"),
    limit: int = Query(100, ge=1, le=1000, description="Number of papers to return")
):
    """Get papers that need AI processing."""
    db = get_db()
    try:
        # Get all papers
        if dataset:
            papers = db.get_papers_by_dataset(dataset)
        else:
            papers = db.get_all_papers()
        
        # Filter unprocessed papers
        unprocessed = []
        for paper in papers:
            needs_processing = False
            
            if missing_summary and (not paper.get("summary") or paper.get("summary").strip() == ""):
                needs_processing = True
            
            if missing_keywords:
                keywords = json.loads(paper.get("keywords", "[]"))
                if not keywords:
                    needs_processing = True
            
            if needs_processing:
                unprocessed.append({
                    "id": paper["id"],
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "source_dataset": paper["source_dataset"],
                    "has_summary": bool(paper.get("summary", "").strip()),
                    "has_keywords": bool(json.loads(paper.get("keywords", "[]"))),
                    "created_at": paper["created_at"]
                })
        
        # Apply limit
        return unprocessed[:limit]
        
    finally:
        db.close()

# WebSocket endpoint for real-time updates
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle specific commands if needed
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve static files from frontend build
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)