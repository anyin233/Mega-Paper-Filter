#!/usr/bin/env python3
"""
Database-driven clustering script for academic papers.
Reads papers from SQLite database, performs clustering, and generates visualization data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
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
from typing import Dict, Any, Optional

from database import PaperDatabase

warnings.filterwarnings('ignore')
console = Console()

class DatabaseClusteringAnalyzer:
    """Clustering analyzer that works with database-stored papers."""
    
    def __init__(self, db_path: str, config: dict = None):
        """Initialize with database connection and configuration."""
        self.db = PaperDatabase(db_path)
        self.config = config or self.get_default_config()
        
        # Analysis results
        self.df = None
        self.features = None
        self.vectorizer = None
        self.cluster_labels = None
        self.cluster_analysis = None
        self.cluster_names = None  # Store LLM-generated cluster names
        self.pca = None
        self.features_2d = None
        
    def get_default_config(self) -> dict:
        """Get default clustering configuration with enhanced keyword weighting."""
        return {
            'tfidf_params': {
                'max_features': 1000,
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'min_df': 2,  # Minimum documents for a term to be included
                'max_df': 0.7,  # Reduced from 0.8 to give more weight to specific terms like keywords
                'sublinear_tf': True,  # Use log-scaled term frequency
                'norm': 'l2'  # L2 normalization to help with keyword emphasis
            },
            'clustering_params': {
                'method': 'kmeans',  # Options: 'kmeans', 'agglomerative', 'dbscan', 'spectral'
                'max_k': 15,
                'random_state': 42,
                'n_init': 10,
                # DBSCAN parameters
                'eps': 0.5,
                'min_samples': 5,
                # Agglomerative parameters
                'linkage': 'ward',  # Options: 'ward', 'complete', 'average', 'single'
                # Spectral parameters
                'assign_labels': 'discretize'  # Options: 'kmeans', 'discretize'
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
        
        # Create combined text for clustering with enhanced keyword weighting
        def create_weighted_text(row):
            """Create text with weighted keywords for better clustering."""
            text_parts = []
            
            # Add title (weight 1x)
            if pd.notna(row['title']):
                text_parts.append(row['title'])
            
            # Add abstract (weight 1x)
            if pd.notna(row['abstract']):
                text_parts.append(row['abstract'])
                
            # Add summary (weight 1x)
            if pd.notna(row['summary']):
                text_parts.append(row['summary'])
            
            # Add keywords with enhanced weight (repeat 3x for higher emphasis)
            keywords_str = row['keywords_str']
            if pd.notna(keywords_str) and keywords_str.strip():
                # Add keywords 3 times to increase their TF-IDF weight
                for _ in range(3):
                    text_parts.append(keywords_str)
                
                # Also add individual keywords as separate terms
                keywords_list = keywords_str.split(', ')
                for keyword in keywords_list:
                    if keyword.strip():
                        # Add each keyword as a standalone term (additional emphasis)
                        text_parts.append(keyword.strip())
            
            return ' '.join(text_parts).strip()
        
        self.df['combined_text'] = self.df.apply(create_weighted_text, axis=1)
        
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
    
    def create_features(self):
        """Create TF-IDF features from combined text."""
        console.print("🔧 Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(**self.config['tfidf_params'])
        self.features = self.vectorizer.fit_transform(self.df['combined_text'])
        
        console.print(f"✅ Created {self.features.shape[1]} features from {len(self.df)} papers")
        return self.features
    
    def _create_clusterer(self, n_clusters: int, method: str = None):
        """Create a clustering algorithm instance based on the specified method."""
        if method is None:
            method = self.config['clustering_params'].get('method', 'kmeans')
        
        params = self.config['clustering_params']
        
        if method == 'kmeans':
            return KMeans(
                n_clusters=n_clusters,
                random_state=params['random_state'],
                n_init=params['n_init']
            )
        elif method == 'agglomerative':
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=params.get('linkage', 'ward')
            )
        elif method == 'dbscan':
            # DBSCAN doesn't use n_clusters, but we include it for interface consistency
            return DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
        elif method == 'spectral':
            return SpectralClustering(
                n_clusters=n_clusters,
                random_state=params['random_state'],
                assign_labels=params.get('assign_labels', 'discretize')
            )
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    def find_optimal_clusters(self) -> tuple:
        """Find optimal number of clusters using silhouette analysis."""
        method = self.config['clustering_params'].get('method', 'kmeans')
        console.print(f"🎯 Finding optimal number of clusters using {method}...")
        
        # DBSCAN doesn't require finding optimal k
        if method == 'dbscan':
            console.print("DBSCAN doesn't require predefined number of clusters")
            # For DBSCAN, we'll return dummy values and let perform_clustering handle it
            return None, [], [], []
        
        max_k = min(self.config['clustering_params']['max_k'], len(self.df) - 1, 20)
        if max_k < 2:
            raise ValueError("Need at least 2 papers for clustering")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        # Scale features for better performance with some algorithms
        if method in ['spectral', 'agglomerative']:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(self.features.toarray())
        else:
            scaled_features = self.features
        
        for k in k_range:
            try:
                clusterer = self._create_clusterer(k, method)
                
                if method in ['spectral', 'agglomerative']:
                    cluster_labels = clusterer.fit_predict(scaled_features)
                else:
                    cluster_labels = clusterer.fit_predict(self.features)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                    if method in ['spectral', 'agglomerative']:
                        score = silhouette_score(scaled_features, cluster_labels)
                    else:
                        score = silhouette_score(self.features, cluster_labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
                
                # Calculate inertia (only for KMeans)
                if method == 'kmeans':
                    inertias.append(clusterer.inertia_)
                else:
                    inertias.append(0)  # Placeholder for other methods
                    
            except Exception as e:
                console.print(f"⚠️ Error with k={k}: {e}")
                silhouette_scores.append(-1)
                inertias.append(0)
        
        # Find optimal k
        if silhouette_scores and max(silhouette_scores) > 0:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            max_silhouette = max(silhouette_scores)
        else:
            optimal_k = min(8, max_k)  # Fallback
            max_silhouette = 0
        
        console.print(f"✅ Optimal clusters: {optimal_k} (silhouette score: {max_silhouette:.3f})")
        
        return optimal_k, inertias, silhouette_scores, k_range
    
    def perform_clustering(self, n_clusters: int = None):
        """Perform clustering using the specified method."""
        method = self.config['clustering_params'].get('method', 'kmeans')
        
        if method == 'dbscan':
            console.print(f"🤖 Performing DBSCAN clustering...")
            clusterer = self._create_clusterer(None, method)  # n_clusters not used for DBSCAN
        else:
            if n_clusters is None:
                n_clusters = 8  # Default fallback
            console.print(f"🤖 Performing {method} clustering with {n_clusters} clusters...")
            clusterer = self._create_clusterer(n_clusters, method)
        
        # Scale features for better performance with some algorithms
        if method in ['spectral', 'agglomerative']:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(self.features.toarray())
            self.cluster_labels = clusterer.fit_predict(scaled_features)
        else:
            self.cluster_labels = clusterer.fit_predict(self.features)
        
        # Handle DBSCAN noise points (labeled as -1)
        if method == 'dbscan':
            n_clusters = len(np.unique(self.cluster_labels))
            n_noise = list(self.cluster_labels).count(-1)
            console.print(f"DBSCAN found {n_clusters - (1 if n_noise > 0 else 0)} clusters and {n_noise} noise points")
            
            # Convert noise points (-1) to a separate cluster for visualization
            if n_noise > 0:
                max_cluster = max(self.cluster_labels)
                self.cluster_labels = np.array([max_cluster + 1 if x == -1 else x for x in self.cluster_labels])
        
        # Add cluster labels to dataframe
        self.df['cluster_id'] = self.cluster_labels
        
        console.print("✅ Clustering complete")
        return clusterer, self.cluster_labels
    
    def create_pca_projection(self):
        """Create PCA projection for visualization."""
        console.print("📊 Creating PCA projection for visualization...")
        
        self.pca = PCA(**self.config['pca_params'])
        self.features_2d = self.pca.fit_transform(self.features.toarray())
        
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
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_papers = self.df[cluster_mask]
            
            # Get cluster center in feature space (keeping for legacy compatibility)
            cluster_features = self.features[cluster_mask]
            cluster_center = cluster_features.mean(axis=0).A1
            
            # Get top features (keeping for reference but not using for LLM)
            top_features_idx = cluster_center.argsort()[-15:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            top_scores = cluster_center[top_features_idx]
            
            # Get sample paper titles
            sample_titles = cluster_papers['title'].head(5).tolist()
            
            # Get sample paper abstracts (NEW: for LLM cluster naming)
            sample_abstracts = []
            if 'summary' in cluster_papers.columns:
                abstracts = cluster_papers['summary'].dropna().head(3).tolist()
                sample_abstracts = [str(abstract) for abstract in abstracts if str(abstract).strip()]
            
            # Get most common keywords from papers in cluster
            all_keywords = []
            for keywords_list in cluster_papers['keywords_parsed']:
                all_keywords.extend(keywords_list)
            
            keyword_freq = pd.Series(all_keywords).value_counts()
            common_keywords = keyword_freq.head(10).to_dict()
            
            self.cluster_analysis[cluster_id] = {
                'size': len(cluster_papers),
                'top_tfidf_features': list(zip(top_features, top_scores)),  # Keep for backward compatibility
                'sample_titles': sample_titles,
                'sample_abstracts': sample_abstracts,  # NEW: actual abstracts for LLM
                'common_keywords': common_keywords,
                'avg_year': cluster_papers['publication_year'].mean() if 'publication_year' in cluster_papers.columns else None
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
        
        # Prepare cluster info using TF-IDF features (more relevant for clustering)
        cluster_info = {}
        for cluster_id, analysis in self.cluster_analysis.items():
            # Use TF-IDF features as main keywords for consistency
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
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'source_database': str(self.db.db_path),
            'dataset_filter': dataset_name,
            'clustering_method': 'K-means',
            'feature_extraction': 'TF-IDF',
            'total_papers_in_db': db_stats['total_papers'],
            'papers_used_for_clustering': len(papers_data),
            'pca_explained_variance': [float(x) for x in self.pca.explained_variance_ratio_],
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
            "[bold blue]CLUSTER ANALYSIS SUMMARY[/bold blue]",
            border_style="blue"
        ))
        
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
            
            # Top TF-IDF features
            console.print("[yellow]Top TF-IDF Features:[/yellow]")
            for feature, score in analysis['top_tfidf_features'][:8]:
                console.print(f"  • {feature}: {score:.3f}")
            
            # Common paper keywords (if available)
            if analysis['common_keywords']:
                console.print("\n[yellow]Common Paper Keywords:[/yellow]")
                for keyword, count in list(analysis['common_keywords'].items())[:5]:
                    console.print(f"  • {keyword}: {count} papers")
            
            # Sample titles
            console.print("\n[yellow]Sample Paper Titles:[/yellow]")
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
        title = f"Academic Papers Clustering Analysis"
        if dataset_name:
            title += f" - {dataset_name}"
        title += f" ({len(self.df)} papers, {len(np.unique(self.cluster_labels))} clusters)"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Cluster scatter plot
        n_clusters = len(np.unique(self.cluster_labels))
        scatter = axes[0, 0].scatter(
            self.features_2d[:, 0], self.features_2d[:, 1],
            c=self.cluster_labels, cmap='tab10', alpha=0.7, s=50
        )
        axes[0, 0].set_title(f'Paper Clusters (K={n_clusters})')
        axes[0, 0].set_xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Cluster size distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        axes[0, 1].bar(cluster_counts.index, cluster_counts.values, 
                      color=plt.cm.tab10(np.linspace(0, 1, len(cluster_counts))))
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Papers')
        
        # 3. Publication year distribution (if available)
        if 'publication_year' in self.df.columns and self.df['publication_year'].notna().sum() > 0:
            year_data = self.df['publication_year'].dropna()
            axes[1, 0].hist(year_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Publication Year Distribution')
            axes[1, 0].set_xlabel('Publication Year')
            axes[1, 0].set_ylabel('Number of Papers')
        else:
            axes[1, 0].text(0.5, 0.5, 'Publication year\ndata not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Publication Year Distribution')
        
        # 4. Dataset distribution (if multiple datasets)
        dataset_counts = self.df['source_dataset'].value_counts()
        if len(dataset_counts) > 1:
            axes[1, 1].pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Papers by Dataset')
        else:
            axes[1, 1].text(0.5, 0.5, f'Single dataset:\n{dataset_counts.index[0]}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Dataset Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"✅ Plots saved to {output_path}")
        
        return fig
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

def main():
    parser = argparse.ArgumentParser(description="Cluster papers from database")
    parser.add_argument('--database', default='papers.db', help='Database file path')
    parser.add_argument('--dataset', help='Dataset name to filter (optional)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--output-prefix', default='clustering', help='Output file prefix')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--show-plots', action='store_true', help='Display plots instead of saving')
    parser.add_argument('--min-papers', type=int, default=5, help='Minimum papers required')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--openai-api-key', help='OpenAI API key for cluster naming')
    parser.add_argument('--openai-base-url', default='https://api.openai.com/v1', help='OpenAI base URL')
    parser.add_argument('--openai-model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--skip-naming', action='store_true', help='Skip LLM-based cluster naming')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.database):
        console.print(f"❌ Database not found: {args.database}")
        console.print("💡 Run parse_paper_db.py first to populate the database")
        return
    
    # Initialize analyzer
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    analyzer = DatabaseClusteringAnalyzer(args.database, config)
    
    try:
        # List datasets if requested
        if args.list_datasets:
            datasets = analyzer.db.get_datasets()
            if not datasets:
                console.print("No datasets found in database")
                return
            
            console.print("[bold blue]Available Datasets:[/bold blue]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dataset", style="cyan")
            table.add_column("Papers", style="yellow")
            table.add_column("Description", style="green")
            table.add_column("Created", style="blue")
            
            for dataset in datasets:
                created = dataset['created_at'].split('T')[0] if dataset['created_at'] else 'Unknown'
                table.add_row(
                    dataset['name'],
                    str(dataset['actual_papers']),
                    dataset['description'] or 'No description',
                    created
                )
            
            console.print(table)
            return
        
        # Display banner
        console.print(Panel.fit(
            f"[bold blue]Database Clustering Analysis[/bold blue]\n"
            f"🗃️  Database: {args.database}\n" +
            (f"📊 Dataset filter: {args.dataset}\n" if args.dataset else "") +
            f"📁 Output directory: {args.output_dir}",
            border_style="blue"
        ))
        
        # Run analysis - create an async function to handle the analysis
        async def run_analysis():
            # Load papers
            df = analyzer.load_papers_from_db(args.dataset, args.min_papers)
            
            # Perform analysis
            analyzer.create_features()
            optimal_k, inertias, silhouette_scores, k_range = analyzer.find_optimal_clusters()
            analyzer.perform_clustering(optimal_k)
            analyzer.create_pca_projection()
            analyzer.analyze_clusters()
            
            # Generate cluster names if OpenAI configuration is provided
            if not args.skip_naming and args.openai_api_key:
                openai_config = {
                    'api_key': args.openai_api_key,
                    'base_url': args.openai_base_url,
                    'model': args.openai_model
                }
                await analyzer.generate_cluster_names(openai_config)
            elif not args.skip_naming:
                console.print("⚠️ OpenAI API key not provided. Use --openai-api-key to enable cluster naming.")
            
            # Generate outputs
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            output_prefix = output_dir / args.output_prefix
            
            # Generate visualization data for web
            json_output = f"{output_prefix}_data.json"
            analyzer.generate_visualization_data(json_output, args.dataset)
            
            # Create plots
            plot_output = f"{output_prefix}_plots.png"
            fig = analyzer.create_visualizations(plot_output, args.dataset)
            
            if args.show_plots:
                plt.show()
            else:
                plt.close(fig)
            
            # Print analysis summary
            analyzer.print_cluster_summary()
            
            # Show completion message
            console.print(f"\n[bold green]✅ Analysis Complete![/bold green]")
            console.print(f"📄 Visualization data: {json_output}")
            console.print(f"📊 Plots: {plot_output}")
            if analyzer.cluster_names:
                console.print("🏷️ Cluster names generated using LLM")
            console.print(f"\n💡 Next step: uv run serve_generic.py --dir {args.output_dir}")
        
        # Run the async analysis
        asyncio.run(run_analysis())
        
    except Exception as e:
        console.print(f"❌ Error during analysis: {e}")
        raise
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()