#!/usr/bin/env python3
"""
Embedding-based clustering script for academic papers.
Uses text embeddings from LLM models with traditional clustering algorithms.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json
import argparse
import os
from pathlib import Path
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import pickle
import time

from database import PaperDatabase

warnings.filterwarnings('ignore')
console = Console()

class EmbeddingClusteringAnalyzer:
    """Clustering analyzer that uses text embeddings from LLM models."""
    
    def __init__(self, db_path: str, config: dict = None):
        """Initialize with database connection and configuration."""
        self.db = PaperDatabase(db_path)
        self.config = config or self.get_default_config()
        
        # Analysis results
        self.df = None
        self.embeddings = None
        self.cluster_labels = None
        self.cluster_analysis = None
        self.cluster_names = None
        self.pca = None
        self.features_2d = None
        
        # Embedding client
        self.embedding_client = None
        self.embedding_config = None
        
    def get_default_config(self) -> dict:
        """Get default clustering configuration."""
        return {
            'embedding_params': {
                'model': 'text-embedding-ada-002',
                'batch_size': 50,
                'max_concurrent': 3,
                'cache_embeddings': True
            },
            'clustering_params': {
                'max_k': 15,
                'random_state': 42,
                'method': 'kmeans',  # 'kmeans', 'dbscan', 'agglomerative'
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5,
                'agglomerative_linkage': 'ward'
            },
            'pca_params': {
                'n_components': 2,
                'random_state': 42
            }
        }
    
    def load_papers_from_db(self, dataset_name: str = None, min_papers: int = 10) -> pd.DataFrame:
        """Load papers from database for analysis."""
        console.print(f"📚 Loading papers from database...")
        
        if dataset_name:
            papers = self.db.get_papers_by_dataset(dataset_name)
            console.print(f"📊 Dataset filter: {dataset_name}")
        else:
            papers = self.db.get_all_papers()
            console.print("📊 Using all papers in database")
        
        if not papers:
            raise ValueError(f"No papers found in database" + 
                           (f" for dataset '{dataset_name}'" if dataset_name else ""))
        
        # Convert to DataFrame
        self.df = pd.DataFrame(papers)
        
        # Parse JSON fields
        self.df['authors_parsed'] = self.df['authors'].apply(
            lambda x: json.loads(x) if x else []
        )
        self.df['keywords_parsed'] = self.df['keywords'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        # Create text fields for analysis
        self.df['authors_str'] = self.df['authors_parsed'].apply(
            lambda x: '; '.join(x) if x else ''
        )
        self.df['keywords_str'] = self.df['keywords_parsed'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        
        # Create combined text for embeddings
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['abstract'].fillna('') + ' ' +
            self.df['summary'].fillna('') + ' ' +
            self.df['keywords_str'].fillna('')
        ).str.strip()
        
        # Filter out papers with insufficient text
        initial_count = len(self.df)
        self.df = self.df[self.df['combined_text'].str.len() > 20]  # Minimum 20 characters
        filtered_count = len(self.df)
        
        if filtered_count < min_papers:
            raise ValueError(f"Insufficient papers for clustering: {filtered_count} (minimum: {min_papers})")
        
        if filtered_count < initial_count:
            console.print(f"⚠️  Filtered out {initial_count - filtered_count} papers with insufficient text")
        
        console.print(f"✅ Loaded {len(self.df)} papers for analysis")
        
        # Display dataset information
        datasets = self.df['source_dataset'].value_counts()
        if len(datasets) > 1:
            console.print(f"📊 Papers by dataset:")
            for dataset, count in datasets.items():
                console.print(f"   • {dataset}: {count} papers")
        
        return self.df
    
    def setup_embedding_client(self, embedding_config: dict):
        """Setup embedding client with configuration."""
        console.print("🔧 Setting up embedding client...")
        
        self.embedding_config = embedding_config
        
        # Import embedding functions
        import sys
        sys.path.append(str(Path(__file__).parent / "src"))
        from openai_api import get_embedding_client
        
        try:
            self.embedding_client = get_embedding_client(
                api_key=embedding_config['api_key'],
                base_url=embedding_config['base_url']
            )
            console.print(f"✅ Embedding client initialized")
            console.print(f"   Model: {embedding_config['model']}")
            console.print(f"   Base URL: {embedding_config['base_url']}")
            console.print(f"   Fallback to OpenAI: {embedding_config.get('fallback_to_openai', False)}")
            
        except Exception as e:
            console.print(f"❌ Failed to initialize embedding client: {e}")
            raise
    
    def get_embedding_cache_path(self, dataset_name: str = None) -> Path:
        """Get path for embedding cache file."""
        cache_dir = Path(".paper_labeler") / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filename based on dataset and model
        model_name = self.embedding_config['model'].replace('/', '_')
        dataset_suffix = f"_{dataset_name}" if dataset_name else "_all"
        cache_file = f"embeddings_{model_name}{dataset_suffix}.pkl"
        
        return cache_dir / cache_file
    
    def save_embedding_cache(self, embeddings: np.ndarray, paper_ids: List[str], dataset_name: str = None):
        """Save embeddings to cache."""
        if not self.config['embedding_params']['cache_embeddings']:
            return
        
        cache_path = self.get_embedding_cache_path(dataset_name)
        cache_data = {
            'embeddings': embeddings,
            'paper_ids': paper_ids,
            'model': self.embedding_config['model'],
            'created_at': time.time()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            console.print(f"💾 Embeddings cached to {cache_path}")
        except Exception as e:
            console.print(f"⚠️ Failed to cache embeddings: {e}")
    
    def load_embedding_cache(self, paper_ids: List[str], dataset_name: str = None) -> Optional[np.ndarray]:
        """Load embeddings from cache if available and valid."""
        if not self.config['embedding_params']['cache_embeddings']:
            return None
        
        cache_path = self.get_embedding_cache_path(dataset_name)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache
            if (cache_data['model'] != self.embedding_config['model'] or
                cache_data['paper_ids'] != paper_ids):
                console.print("⚠️ Embedding cache outdated, will regenerate")
                return None
            
            # Check if cache is recent (within 7 days)
            cache_age_days = (time.time() - cache_data['created_at']) / (24 * 3600)
            if cache_age_days > 7:
                console.print(f"⚠️ Embedding cache is {cache_age_days:.1f} days old, will regenerate")
                return None
            
            console.print(f"✅ Loaded embeddings from cache ({cache_age_days:.1f} days old)")
            return cache_data['embeddings']
            
        except Exception as e:
            console.print(f"⚠️ Failed to load embedding cache: {e}")
            return None
    
    def create_embeddings(self) -> np.ndarray:
        """Create embeddings for all papers."""
        console.print("🔮 Creating text embeddings...")
        
        if self.embedding_client is None:
            raise ValueError("Embedding client not initialized. Call setup_embedding_client first.")
        
        paper_ids = self.df['paper_id'].tolist()
        texts = self.df['combined_text'].tolist()
        
        # Try to load from cache
        cached_embeddings = self.load_embedding_cache(paper_ids)
        if cached_embeddings is not None:
            self.embeddings = cached_embeddings
            console.print(f"✅ Using cached embeddings: {self.embeddings.shape}")
            return self.embeddings
        
        # Generate new embeddings
        console.print(f"🚀 Generating embeddings for {len(texts)} papers...")
        
        import sys
        sys.path.append(str(Path(__file__).parent / "src"))
        from openai_api import get_batch_text_embeddings
        
        try:
            embeddings_list = get_batch_text_embeddings(
                client=self.embedding_client,
                texts=texts,
                model=self.embedding_config['model'],
                batch_size=self.config['embedding_params']['batch_size']
            )
            
            self.embeddings = np.array(embeddings_list)
            console.print(f"✅ Generated embeddings: {self.embeddings.shape}")
            
            # Cache the embeddings
            dataset_name = self.config.get('dataset_name')
            self.save_embedding_cache(self.embeddings, paper_ids, dataset_name)
            
        except Exception as e:
            console.print(f"❌ Failed to generate embeddings: {e}")
            raise
        
        return self.embeddings
    
    def find_optimal_clusters(self) -> Tuple[int, List[float], List[float], List[int]]:
        """Find optimal number of clusters using silhouette analysis."""
        console.print("🎯 Finding optimal number of clusters...")
        
        method = self.config['clustering_params']['method']
        max_k = min(self.config['clustering_params']['max_k'], len(self.df) - 1, 20)
        
        if method == 'dbscan':
            # For DBSCAN, we don't need to find optimal k
            return self._find_optimal_dbscan_params()
        elif method == 'agglomerative':
            # For Agglomerative clustering, find optimal number of clusters
            return self._find_optimal_agglomerative(max_k)
        else:
            # Default K-means
            return self._find_optimal_kmeans(max_k)
    
    def _find_optimal_kmeans(self, max_k: int) -> Tuple[int, List[float], List[float], List[int]]:
        """Find optimal K for K-means clustering."""
        if max_k < 2:
            raise ValueError("Need at least 2 papers for clustering")
        
        inertias = []
        silhouette_scores = []
        k_range = list(range(2, max_k + 1))
        
        console.print(f"🔍 Testing K-means with k from 2 to {max_k}...")
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config['clustering_params']['random_state'],
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.embeddings, cluster_labels))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        
        console.print(f"✅ Optimal K-means clusters: {optimal_k} (silhouette score: {max_silhouette:.3f})")
        
        return optimal_k, inertias, silhouette_scores, k_range
    
    def _find_optimal_agglomerative(self, max_k: int) -> Tuple[int, List[float], List[float], List[int]]:
        """Find optimal number of clusters for Agglomerative clustering."""
        if max_k < 2:
            raise ValueError("Need at least 2 papers for clustering")
        
        inertias = []  # Not applicable for agglomerative, but keep for consistency
        silhouette_scores = []
        k_range = list(range(2, max_k + 1))
        
        console.print(f"🔍 Testing Agglomerative clustering with k from 2 to {max_k}...")
        
        for k in k_range:
            agg = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.config['clustering_params']['agglomerative_linkage']
            )
            cluster_labels = agg.fit_predict(self.embeddings)
            
            inertias.append(0.0)  # Placeholder
            silhouette_scores.append(silhouette_score(self.embeddings, cluster_labels))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        
        console.print(f"✅ Optimal Agglomerative clusters: {optimal_k} (silhouette score: {max_silhouette:.3f})")
        
        return optimal_k, inertias, silhouette_scores, k_range
    
    def _find_optimal_dbscan_params(self) -> Tuple[int, List[float], List[float], List[int]]:
        """Find optimal parameters for DBSCAN clustering."""
        console.print("🔍 Testing DBSCAN parameters...")
        
        eps = self.config['clustering_params']['dbscan_eps']
        min_samples = self.config['clustering_params']['dbscan_min_samples']
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters < 2:
            console.print("⚠️ DBSCAN found less than 2 clusters, adjusting parameters...")
            # Try with smaller eps
            eps = eps * 0.7
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters >= 2:
            # Filter out noise points for silhouette calculation
            non_noise_mask = cluster_labels != -1
            if non_noise_mask.sum() >= 2:
                silhouette_avg = silhouette_score(
                    self.embeddings[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        console.print(f"✅ DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        console.print(f"   Silhouette score: {silhouette_avg:.3f}")
        
        # Return format compatible with other methods
        return n_clusters, [0.0], [silhouette_avg], [n_clusters]
    
    def perform_clustering(self, n_clusters: int):
        """Perform clustering using the specified method."""
        method = self.config['clustering_params']['method']
        console.print(f"🤖 Performing {method.upper()} clustering...")
        
        if method == 'dbscan':
            return self._perform_dbscan()
        elif method == 'agglomerative':
            return self._perform_agglomerative(n_clusters)
        else:
            return self._perform_kmeans(n_clusters)
    
    def _perform_kmeans(self, n_clusters: int):
        """Perform K-means clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['clustering_params']['random_state'],
            n_init=10
        )
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Add cluster labels to dataframe
        self.df['cluster_id'] = self.cluster_labels
        
        console.print(f"✅ K-means clustering complete ({n_clusters} clusters)")
        return kmeans, self.cluster_labels
    
    def _perform_agglomerative(self, n_clusters: int):
        """Perform Agglomerative clustering."""
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config['clustering_params']['agglomerative_linkage']
        )
        self.cluster_labels = agg.fit_predict(self.embeddings)
        
        # Add cluster labels to dataframe
        self.df['cluster_id'] = self.cluster_labels
        
        console.print(f"✅ Agglomerative clustering complete ({n_clusters} clusters)")
        return agg, self.cluster_labels
    
    def _perform_dbscan(self):
        """Perform DBSCAN clustering."""
        eps = self.config['clustering_params']['dbscan_eps']
        min_samples = self.config['clustering_params']['dbscan_min_samples']
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.embeddings)
        
        # Add cluster labels to dataframe
        self.df['cluster_id'] = self.cluster_labels
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        console.print(f"✅ DBSCAN clustering complete ({n_clusters} clusters, {n_noise} noise points)")
        return dbscan, self.cluster_labels
    
    def create_pca_projection(self):
        """Create PCA projection for visualization."""
        console.print("📊 Creating PCA projection for visualization...")
        
        self.pca = PCA(**self.config['pca_params'])
        self.features_2d = self.pca.fit_transform(self.embeddings)
        
        # Add PCA coordinates to dataframe
        self.df['pca_x'] = self.features_2d[:, 0]
        self.df['pca_y'] = self.features_2d[:, 1]
        
        explained_var = self.pca.explained_variance_ratio_
        console.print(f"✅ PCA complete (explained variance: {explained_var[0]:.1%} + {explained_var[1]:.1%} = {sum(explained_var):.1%})")
        
        return self.pca, self.features_2d
    
    def analyze_clusters(self):
        """Analyze cluster characteristics."""
        console.print("🔍 Analyzing cluster characteristics...")
        
        self.cluster_analysis = {}
        
        unique_labels = np.unique(self.cluster_labels)
        
        for cluster_id in unique_labels:
            # Skip noise points in DBSCAN (cluster_id = -1)
            if cluster_id == -1:
                continue
                
            cluster_mask = self.cluster_labels == cluster_id
            cluster_papers = self.df[cluster_mask]
            
            # Get cluster center in embedding space
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_center = cluster_embeddings.mean(axis=0)
            
            # Calculate distances from center to find representative papers
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            representative_indices = np.argsort(distances)[:5]  # 5 most representative papers
            
            # Get sample paper titles (most representative ones)
            sample_titles = cluster_papers.iloc[representative_indices]['title'].tolist()
            
            # Get sample paper abstracts for LLM cluster naming
            sample_abstracts = []
            if 'summary' in cluster_papers.columns:
                abstracts = cluster_papers.iloc[representative_indices]['summary'].dropna().head(3).tolist()
                sample_abstracts = [str(abstract) for abstract in abstracts if str(abstract).strip()]
            
            # Get most common keywords from papers in cluster
            all_keywords = []
            for keywords_list in cluster_papers['keywords_parsed']:
                all_keywords.extend(keywords_list)
            
            keyword_freq = pd.Series(all_keywords).value_counts()
            common_keywords = keyword_freq.head(10).to_dict()
            
            # Generate synthetic "top features" for compatibility
            # Use most common words from combined text
            all_text = ' '.join(cluster_papers['combined_text'].fillna(''))
            words = all_text.lower().split()
            word_freq = pd.Series(words).value_counts()
            
            # Filter out common stop words and generate top features
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            filtered_words = word_freq[~word_freq.index.isin(stop_words)]
            top_features = list(zip(filtered_words.head(15).index, filtered_words.head(15).values / len(words)))
            
            self.cluster_analysis[cluster_id] = {
                'size': len(cluster_papers),
                'top_tfidf_features': top_features,  # Synthetic features for compatibility
                'sample_titles': sample_titles,
                'sample_abstracts': sample_abstracts,
                'common_keywords': common_keywords,
                'avg_year': cluster_papers['publication_year'].mean() if 'publication_year' in cluster_papers.columns else None,
                'cluster_center': cluster_center.tolist(),  # Store for potential future use
                'representative_paper_indices': representative_indices.tolist()
            }
        
        console.print(f"✅ Analyzed {len(self.cluster_analysis)} clusters")
        return self.cluster_analysis
    
    async def generate_cluster_names(self, openai_config: Optional[Dict[str, Any]] = None):
        """Generate LLM-based names for clusters."""
        if not self.cluster_analysis:
            console.print("❌ No cluster analysis available. Run analyze_clusters() first.")
            return
            
        if not openai_config or not openai_config.get('api_key'):
            console.print("⚠️ OpenAI configuration not provided. Skipping cluster naming.")
            return
        
        console.print("🤖 Generating cluster names using LLM...")
        
        try:
            # Import OpenAI functions
            import sys
            sys.path.append(str(Path(__file__).parent / "src"))
            from openai_api import get_async_openai_client, get_cluster_name
            
            # Create async client
            client = get_async_openai_client(openai_config["api_key"], openai_config.get("base_url", "https://api.openai.com/v1"))
            model = openai_config.get("model", "gpt-4o-mini")
            
            self.cluster_names = {}
            
            for cluster_id, analysis in self.cluster_analysis.items():
                try:
                    console.print(f"  🏷️ Naming cluster {cluster_id}...")
                    
                    # Get cluster name from LLM
                    naming_response = await get_cluster_name(client, analysis, model)
                    
                    # Use robust JSON parsing instead of direct json.loads
                    import sys
                    sys.path.append(str(Path(__file__).parent / "src"))
                    from openai_api import safe_json_parse
                    
                    naming_data = safe_json_parse(naming_response, fallback_dict={
                        "name": f"Cluster {cluster_id}",
                        "description": "Cluster naming failed due to JSON parsing error"
                    })
                    
                    self.cluster_names[cluster_id] = {
                        'name': naming_data.get('name', f'Cluster {cluster_id}'),
                        'description': naming_data.get('description', 'No description available')
                    }
                    
                    console.print(f"    ✅ {self.cluster_names[cluster_id]['name']}")
                    
                except Exception as e:
                    console.print(f"    ❌ Failed to name cluster {cluster_id}: {e}")
                    self.cluster_names[cluster_id] = {
                        'name': f'Cluster {cluster_id}',
                        'description': 'Automated naming failed'
                    }
            
            console.print(f"✅ Generated names for {len(self.cluster_names)} clusters")
            
        except Exception as e:
            console.print(f"❌ Error during cluster naming: {e}")
            # Fallback to generic names
            self.cluster_names = {}
            for cluster_id in self.cluster_analysis.keys():
                self.cluster_names[cluster_id] = {
                    'name': f'Cluster {cluster_id}',
                    'description': 'LLM naming unavailable'
                }
        
        return self.cluster_names
    
    def generate_visualization_data(self, output_path: str, dataset_name: str = None):
        """Generate JSON data for web visualization."""
        console.print("📄 Generating visualization data...")
        
        # Prepare papers data for web interface
        papers_data = []
        for _, paper in self.df.iterrows():
            paper_dict = {
                'Key': str(paper.get('paper_id', paper.get('id', ''))),
                'Title': paper.get('title', ''),
                'Author': paper.get('authors_str', ''),
                'summary': paper.get('summary', ''),
                'keywords': paper.get('keywords_str', ''),
                'cluster_id': int(paper['cluster_id']),
                'pca_x': float(paper['pca_x']),
                'pca_y': float(paper['pca_y']),
                'Publication Year': paper.get('publication_year', ''),
                'DOI': paper.get('doi', ''),
                'Url': paper.get('url', ''),
                'Venue': paper.get('venue', ''),
                'Abstract': paper.get('abstract', '')
            }
            papers_data.append(paper_dict)
        
        # Prepare cluster info
        cluster_info = {}
        for cluster_id, analysis in self.cluster_analysis.items():
            # Use synthetic features as main keywords for consistency
            top_keywords = [(kw, float(score)) for kw, score in analysis['top_tfidf_features'][:10]]
            
            cluster_data = {
                'size': analysis['size'],
                'top_keywords': top_keywords,
                'sample_titles': analysis['sample_titles'],
                'common_paper_keywords': analysis['common_keywords']
            }
            
            # Add LLM-generated names if available
            if self.cluster_names and cluster_id in self.cluster_names:
                cluster_data['name'] = self.cluster_names[cluster_id]['name']
                cluster_data['description'] = self.cluster_names[cluster_id]['description']
            else:
                cluster_data['name'] = f'Cluster {cluster_id}'
                cluster_data['description'] = f'Cluster {cluster_id} with {analysis["size"]} papers'
            
            cluster_info[str(cluster_id)] = cluster_data
        
        # Create metadata
        db_stats = self.db.get_statistics()
        clustering_method = self.config['clustering_params']['method']
        
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'source_database': str(self.db.db_path),
            'dataset_filter': dataset_name,
            'clustering_method': f'{clustering_method.upper()} + Embeddings',
            'feature_extraction': f'LLM Embeddings ({self.embedding_config["model"]})',
            'total_papers_in_db': db_stats['total_papers'],
            'papers_used_for_clustering': len(papers_data),
            'pca_explained_variance': [float(x) for x in self.pca.explained_variance_ratio_],
            'embedding_model': self.embedding_config['model'],
            'clustering_config': self.config
        }
        
        # Create complete JSON structure
        json_data = {
            'papers': papers_data,
            'cluster_info': cluster_info,
            'total_papers': len(papers_data),
            'total_clusters': len(cluster_info),
            'metadata': metadata
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"✅ Visualization data saved to {output_path}")
        return json_data
    
    def print_cluster_summary(self):
        """Print detailed cluster analysis summary."""
        console.print(Panel.fit(
            "[bold blue]EMBEDDING-BASED CLUSTER ANALYSIS SUMMARY[/bold blue]",
            border_style="blue"
        ))
        
        method = self.config['clustering_params']['method'].upper()
        model = self.embedding_config['model']
        console.print(f"\n[bold cyan]Method:[/bold cyan] {method} + {model} Embeddings")
        
        for cluster_id, analysis in self.cluster_analysis.items():
            # Display cluster name if available
            cluster_title = f"[bold cyan]Cluster {cluster_id}[/bold cyan]"
            if self.cluster_names and cluster_id in self.cluster_names:
                cluster_name = self.cluster_names[cluster_id]['name']
                cluster_desc = self.cluster_names[cluster_id]['description']
                cluster_title = f"[bold cyan]Cluster {cluster_id}: {cluster_name}[/bold cyan]"
                console.print(f"\n{cluster_title} ({analysis['size']} papers)")
                console.print(f"[italic]{cluster_desc}[/italic]")
            else:
                console.print(f"\n{cluster_title} ({analysis['size']} papers)")
            
            console.print("─" * 50)
            
            # Top features (most common words)
            console.print("[yellow]Top Features:[/yellow]")
            for feature, score in analysis['top_tfidf_features'][:8]:
                console.print(f"  • {feature}: {score:.3f}")
            
            # Common paper keywords (if available)
            if analysis['common_keywords']:
                console.print("\n[yellow]Common Paper Keywords:[/yellow]")
                for keyword, count in list(analysis['common_keywords'].items())[:5]:
                    console.print(f"  • {keyword}: {count} papers")
            
            # Sample titles (most representative)
            console.print("\n[yellow]Representative Paper Titles:[/yellow]")
            for i, title in enumerate(analysis['sample_titles'][:3], 1):
                console.print(f"  {i}. {title}")
            
            # Average year if available
            if analysis['avg_year'] and not pd.isna(analysis['avg_year']):
                console.print(f"\n[yellow]Average Publication Year:[/yellow] {analysis['avg_year']:.1f}")
    
    def create_visualizations(self, output_path: str, dataset_name: str = None):
        """Create matplotlib visualizations."""
        console.print("📊 Creating visualization plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main title
        method = self.config['clustering_params']['method'].upper()
        model = self.embedding_config['model']
        title = f"Embedding-based Clustering Analysis ({method} + {model})"
        if dataset_name:
            title += f" - {dataset_name}"
        title += f" ({len(self.df)} papers, {len(np.unique(self.cluster_labels))} clusters)"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Cluster scatter plot
        n_clusters = len(np.unique(self.cluster_labels))
        # Handle noise points in DBSCAN
        colors = self.cluster_labels.copy()
        if -1 in colors:  # DBSCAN noise points
            colors = colors + 1  # Shift so -1 becomes 0 (noise), others shift up
        
        scatter = axes[0, 0].scatter(
            self.features_2d[:, 0], self.features_2d[:, 1],
            c=colors, cmap='tab10', alpha=0.7, s=50
        )
        axes[0, 0].set_title(f'Paper Clusters ({method})')
        axes[0, 0].set_xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Cluster size distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        # Filter out noise points for visualization
        cluster_counts = cluster_counts[cluster_counts.index != -1]
        
        if len(cluster_counts) > 0:
            axes[0, 1].bar(cluster_counts.index, cluster_counts.values, 
                          color=plt.cm.tab10(np.linspace(0, 1, len(cluster_counts))))
            axes[0, 1].set_title('Cluster Size Distribution')
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Number of Papers')
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid clusters found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Cluster Size Distribution')
        
        # 3. Publication year distribution (if available)
        if 'publication_year' in self.df.columns and self.df['publication_year'].notna().sum() > 0:
            year_data = self.df['publication_year'].dropna()
            axes[1, 0].hist(year_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Publication Year Distribution')
            axes[1, 0].set_xlabel('Publication Year')
            axes[1, 0].set_ylabel('Number of Papers')
        else:
            axes[1, 0].text(0.5, 0.5, 'Publication year\\ndata not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Publication Year Distribution')
        
        # 4. Embedding visualization (embedding space projection)
        # Show first 2 dimensions of embeddings (before PCA)
        axes[1, 1].scatter(
            self.embeddings[:, 0], self.embeddings[:, 1],
            c=colors, cmap='tab10', alpha=0.7, s=30
        )
        axes[1, 1].set_title('Raw Embedding Space (Dims 0-1)')
        axes[1, 1].set_xlabel('Embedding Dimension 0')
        axes[1, 1].set_ylabel('Embedding Dimension 1')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"✅ Plots saved to {output_path}")
        
        return fig
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

# Main execution function would go here, similar to cluster_from_db.py
# but I'll keep it separate for now to focus on the core functionality