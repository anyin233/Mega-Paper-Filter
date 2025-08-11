#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

class PaperClusteringAnalyzer:
    def __init__(self, config=None):
        """
        Initialize the Paper Clustering Analyzer with configurable parameters.
        
        Args:
            config (dict): Configuration dictionary with clustering parameters
        """
        self.config = config or self.get_default_config()
        self.df = None
        self.features = None
        self.vectorizer = None
        self.cluster_labels = None
        self.cluster_analysis = None
        self.pca = None
        self.features_2d = None
        
    def get_default_config(self):
        """Get default configuration parameters"""
        return {
            'csv_columns': {
                'title': 'Title',
                'author': 'Author', 
                'summary': 'summary',
                'keywords': 'keywords',
                'key': 'Key',
                'year': 'Publication Year',
                'doi': 'DOI',
                'url': 'Url'
            },
            'tfidf_params': {
                'max_features': 1000,
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8
            },
            'clustering_params': {
                'max_k': 15,
                'random_state': 42,
                'n_init': 10
            },
            'pca_params': {
                'n_components': 2,
                'random_state': 42
            }
        }

    def load_data(self, csv_path):
        """Load papers dataset from CSV"""
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} papers")
        
        # Validate required columns exist
        required_cols = ['title', 'summary', 'keywords']
        missing_cols = []
        
        for req_col in required_cols:
            config_col = self.config['csv_columns'].get(req_col)
            if config_col not in self.df.columns:
                missing_cols.append(f"{req_col} (expected column: {config_col})")
        
        if missing_cols:
            print(f"Warning: Missing columns: {', '.join(missing_cols)}")
            print("Available columns:", list(self.df.columns))
        
        return self.df

    def preprocess_text(self):
        """Preprocess and combine text features"""
        print("Preprocessing text data...")
        
        # Get column mappings
        cols = self.config['csv_columns']
        
        # Handle missing values for text columns
        for text_col in ['title', 'summary', 'keywords']:
            col_name = cols.get(text_col, text_col)
            if col_name in self.df.columns:
                self.df[col_name] = self.df[col_name].fillna('')
            else:
                # Create empty column if missing
                self.df[col_name] = ''
        
        # Combine text features
        title_col = cols.get('title', 'title')
        summary_col = cols.get('summary', 'summary') 
        keywords_col = cols.get('keywords', 'keywords')
        
        self.df['combined_text'] = (
            self.df[title_col].astype(str) + ' ' + 
            self.df[summary_col].astype(str) + ' ' + 
            self.df[keywords_col].astype(str)
        )
        
        # Remove empty rows
        initial_count = len(self.df)
        self.df = self.df[self.df['combined_text'].str.strip() != '']
        print(f"After preprocessing: {len(self.df)} papers (removed {initial_count - len(self.df)} empty entries)")
        
        return self.df

    def create_features(self):
        """Create TF-IDF features from the combined text"""
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(**self.config['tfidf_params'])
        self.features = self.vectorizer.fit_transform(self.df['combined_text'])
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Created {self.features.shape[1]} features")
        return self.features, self.vectorizer, feature_names

    def find_optimal_clusters(self):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("Finding optimal number of clusters...")
        
        max_k = min(self.config['clustering_params']['max_k'], len(self.df) - 1)
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.config['clustering_params']['random_state'],
                n_init=self.config['clustering_params']['n_init']
            )
            kmeans.fit(self.features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features, kmeans.labels_))
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k, inertias, silhouette_scores, k_range

    def perform_clustering(self, n_clusters):
        """Perform K-means clustering"""
        print("Performing clustering...")
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.config['clustering_params']['random_state'],
            n_init=self.config['clustering_params']['n_init']
        )
        self.cluster_labels = kmeans.fit_predict(self.features)
        
        return kmeans, self.cluster_labels

    def create_pca_projection(self):
        """Create PCA projection for visualization"""
        print("Creating PCA projection...")
        
        self.pca = PCA(**self.config['pca_params'])
        self.features_2d = self.pca.fit_transform(self.features.toarray())
        
        return self.pca, self.features_2d

    def analyze_clusters(self):
        """Analyze and describe each cluster"""
        print("Analyzing clusters...")
        
        self.cluster_analysis = {}
        cols = self.config['csv_columns']
        title_col = cols.get('title', 'title')
        
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_papers = self.df[cluster_mask]
            
            # Get cluster center in feature space
            cluster_features = self.features[cluster_mask]
            cluster_center = cluster_features.mean(axis=0).A1
            
            # Get top features (keywords) for this cluster
            feature_names = self.vectorizer.get_feature_names_out()
            top_features_idx = cluster_center.argsort()[-10:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            top_scores = cluster_center[top_features_idx]
            
            # Get sample paper titles
            sample_titles = cluster_papers[title_col].head(3).tolist()
            
            self.cluster_analysis[cluster_id] = {
                'size': len(cluster_papers),
                'top_keywords': list(zip(top_features, top_scores)),
                'sample_titles': sample_titles
            }
        
        return self.cluster_analysis

    def generate_json_output(self, output_path, source_file):
        """Generate JSON data for web visualization"""
        print("Generating JSON data for web visualization...")
        
        # Add coordinates and cluster info to dataframe
        df_with_coords = self.df.copy()
        df_with_coords['cluster_id'] = self.cluster_labels
        df_with_coords['pca_x'] = self.features_2d[:, 0]
        df_with_coords['pca_y'] = self.features_2d[:, 1]
        
        cols = self.config['csv_columns']
        
        # Prepare papers data
        papers_data = []
        for _, paper in df_with_coords.iterrows():
            paper_dict = {
                'Key': paper.get(cols.get('key', 'Key'), ''),
                'Title': paper.get(cols.get('title', 'Title'), ''),
                'Author': paper.get(cols.get('author', 'Author'), ''),
                'summary': paper.get(cols.get('summary', 'summary'), ''),
                'keywords': paper.get(cols.get('keywords', 'keywords'), ''),
                'cluster_id': int(paper['cluster_id']),
                'pca_x': float(paper['pca_x']),
                'pca_y': float(paper['pca_y']),
                'Publication Year': paper.get(cols.get('year', 'Publication Year'), ''),
                'DOI': paper.get(cols.get('doi', 'DOI'), ''),
                'Url': paper.get(cols.get('url', 'Url'), '')
            }
            papers_data.append(paper_dict)
        
        # Prepare cluster info
        cluster_info = {}
        for cluster_id, analysis in self.cluster_analysis.items():
            cluster_info[str(cluster_id)] = {
                'size': analysis['size'],
                'top_keywords': [(keyword, float(score)) for keyword, score in analysis['top_keywords']],
                'sample_titles': analysis['sample_titles']
            }
        
        # Create the complete JSON structure
        json_data = {
            'papers': papers_data,
            'cluster_info': cluster_info,
            'total_papers': len(papers_data),
            'total_clusters': len(cluster_info),
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'source_file': os.path.basename(source_file),
                'clustering_method': 'K-means',
                'feature_extraction': 'TF-IDF',
                'pca_explained_variance': [float(x) for x in self.pca.explained_variance_ratio_] if self.pca else []
            }
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON data saved to {output_path}")
        return json_data

    def create_visualizations(self, n_clusters):
        """Create clustering visualizations"""
        print("Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Academic Papers Clustering Analysis ({os.path.basename(self.config.get("source_file", "Dataset"))})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Scatter plot of clusters
        scatter = axes[0, 0].scatter(self.features_2d[:, 0], self.features_2d[:, 1], 
                                    c=self.cluster_labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title(f'Paper Clusters (K={n_clusters})')
        axes[0, 0].set_xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Cluster size distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        axes[0, 1].bar(cluster_counts.index, cluster_counts.values)
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Papers')
        
        # 3. Remove unused subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig

    def print_cluster_summary(self):
        """Print a summary of each cluster"""
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS SUMMARY")
        print("="*80)
        
        for cluster_id, analysis in self.cluster_analysis.items():
            print(f"\nCluster {cluster_id} ({analysis['size']} papers):")
            print("-" * 50)
            
            print("Top Keywords:")
            for keyword, score in analysis['top_keywords'][:5]:
                print(f"  • {keyword}: {score:.3f}")
            
            print("\nSample Paper Titles:")
            for i, title in enumerate(analysis['sample_titles'], 1):
                print(f"  {i}. {title}")
            print()

    def save_csv_results(self, output_path):
        """Save clustering results to CSV"""
        result_df = self.df.copy()
        result_df['cluster_id'] = self.cluster_labels
        result_df['pca_x'] = self.features_2d[:, 0]
        result_df['pca_y'] = self.features_2d[:, 1]
        result_df.to_csv(output_path, index=False)
        print(f"CSV results saved to {output_path}")

def create_config_from_args(args):
    """Create configuration from command line arguments"""
    config = PaperClusteringAnalyzer().get_default_config()
    
    if args.title_col:
        config['csv_columns']['title'] = args.title_col
    if args.author_col:
        config['csv_columns']['author'] = args.author_col
    if args.summary_col:
        config['csv_columns']['summary'] = args.summary_col
    if args.keywords_col:
        config['csv_columns']['keywords'] = args.keywords_col
    if args.key_col:
        config['csv_columns']['key'] = args.key_col
    
    if args.max_features:
        config['tfidf_params']['max_features'] = args.max_features
    if args.max_clusters:
        config['clustering_params']['max_k'] = args.max_clusters
    
    config['source_file'] = args.input_csv
    return config

def main():
    parser = argparse.ArgumentParser(description='Generalized Academic Paper Clustering Analysis')
    
    # Input/Output arguments
    parser.add_argument('input_csv', help='Path to input CSV file containing papers')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--output-prefix', default='clustering', help='Prefix for output files')
    
    # CSV column mapping arguments
    parser.add_argument('--title-col', help='Name of title column in CSV')
    parser.add_argument('--author-col', help='Name of author column in CSV')
    parser.add_argument('--summary-col', help='Name of summary/abstract column in CSV')
    parser.add_argument('--keywords-col', help='Name of keywords column in CSV')
    parser.add_argument('--key-col', help='Name of unique key/ID column in CSV')
    
    # Clustering parameters
    parser.add_argument('--max-features', type=int, help='Maximum number of TF-IDF features')
    parser.add_argument('--max-clusters', type=int, help='Maximum number of clusters to try')
    parser.add_argument('--show-plots', action='store_true', help='Display matplotlib plots')
    
    # Configuration file
    parser.add_argument('--config', help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        config['source_file'] = args.input_csv
        # Convert JSON arrays to tuples where needed
        if 'ngram_range' in config.get('tfidf_params', {}):
            config['tfidf_params']['ngram_range'] = tuple(config['tfidf_params']['ngram_range'])
    else:
        config = create_config_from_args(args)
    
    # Initialize analyzer
    analyzer = PaperClusteringAnalyzer(config)
    
    # Run analysis pipeline
    analyzer.load_data(args.input_csv)
    analyzer.preprocess_text()
    analyzer.create_features()
    
    # Find optimal clusters and perform clustering
    optimal_k, inertias, silhouette_scores, k_range = analyzer.find_optimal_clusters()
    kmeans, cluster_labels = analyzer.perform_clustering(optimal_k)
    
    # Create PCA projection and analyze clusters
    analyzer.create_pca_projection()
    analyzer.analyze_clusters()
    
    # Generate outputs
    output_prefix = os.path.join(args.output_dir, args.output_prefix)
    
    # Save JSON for web visualization
    json_output = f"{output_prefix}_data.json"
    analyzer.generate_json_output(json_output, args.input_csv)
    
    # Save CSV results
    csv_output = f"{output_prefix}_results.csv"
    analyzer.save_csv_results(csv_output)
    
    # Print summary
    analyzer.print_cluster_summary()
    
    # Create and optionally show visualizations
    fig = analyzer.create_visualizations(optimal_k)
    
    if args.show_plots:
        plt.show()
    else:
        plot_output = f"{output_prefix}_plots.png"
        fig.savefig(plot_output, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_output}")
        plt.close()
    
    print(f"\nAnalysis complete! Generated files:")
    print(f"  • {json_output} - Data for web visualization")
    print(f"  • {csv_output} - CSV with cluster assignments")
    print(f"  • {plot_output if not args.show_plots else 'Interactive plots displayed'} - Visualization plots")

if __name__ == "__main__":
    main()