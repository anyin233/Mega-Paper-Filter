#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    """Load the ASPLOS papers dataset"""
    df = pd.read_csv(csv_path)
    return df

def preprocess_text(df):
    """Preprocess and combine text features"""
    # Handle missing values
    df['Title'] = df['Title'].fillna('')
    df['summary'] = df['summary'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    
    # Combine title, summary, and keywords into a single text feature
    df['combined_text'] = (df['Title'] + ' ' + 
                          df['summary'] + ' ' + 
                          df['keywords'])
    
    # Remove empty rows
    df = df[df['combined_text'].str.strip() != '']
    
    return df

def create_features(df):
    """Create TF-IDF features from the combined text"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    features = vectorizer.fit_transform(df['combined_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    return features, vectorizer, feature_names

def find_optimal_clusters(features, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, kmeans.labels_))
    
    # Find optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, inertias, silhouette_scores, k_range

def perform_clustering(features, n_clusters):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    return kmeans, cluster_labels

def visualize_clusters(features, cluster_labels, df, n_clusters):
    """Create visualizations of the clusters"""
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features.toarray())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ASPLOS Papers Clustering Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot of clusters
    scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.6)
    axes[0, 0].set_title(f'Paper Clusters (K={n_clusters})')
    axes[0, 0].set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Cluster size distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    axes[0, 1].bar(cluster_counts.index, cluster_counts.values)
    axes[0, 1].set_title('Cluster Size Distribution')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Number of Papers')
    
    # 3. Top keywords per cluster
    # This will be filled by the analyze_clusters function
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Top Keywords per Cluster')
    
    # 4. Sample titles per cluster
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Sample Paper Titles per Cluster')
    
    plt.tight_layout()
    return fig, pca, features_2d

def analyze_clusters(df, cluster_labels, vectorizer, features):
    """Analyze and describe each cluster"""
    cluster_analysis = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_papers = df[cluster_mask]
        
        # Get cluster center in feature space
        cluster_features = features[cluster_mask]
        cluster_center = cluster_features.mean(axis=0).A1
        
        # Get top features (keywords) for this cluster
        feature_names = vectorizer.get_feature_names_out()
        top_features_idx = cluster_center.argsort()[-10:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        top_scores = cluster_center[top_features_idx]
        
        # Get sample paper titles
        sample_titles = cluster_papers['Title'].head(3).tolist()
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_papers),
            'top_keywords': list(zip(top_features, top_scores)),
            'sample_titles': sample_titles
        }
    
    return cluster_analysis

def save_results(df, cluster_labels, cluster_analysis, output_path):
    """Save clustering results to CSV"""
    result_df = df.copy()
    result_df['cluster_id'] = cluster_labels
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def print_cluster_summary(cluster_analysis):
    """Print a summary of each cluster"""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*80)
    
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\nCluster {cluster_id} ({analysis['size']} papers):")
        print("-" * 50)
        
        print("Top Keywords:")
        for keyword, score in analysis['top_keywords'][:5]:
            print(f"  • {keyword}: {score:.3f}")
        
        print("\nSample Paper Titles:")
        for i, title in enumerate(analysis['sample_titles'], 1):
            print(f"  {i}. {title}")
        print()

def plot_optimization_metrics(inertias, silhouette_scores, k_range, optimal_k):
    """Plot elbow curve and silhouette scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.axvline(x=optimal_k, color='g', linestyle='--', 
                label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_json_for_web(df, cluster_labels, cluster_analysis, features_2d, output_path):
    """Generate JSON data for the web visualization"""
    # Add PCA coordinates to dataframe
    df_with_coords = df.copy()
    df_with_coords['cluster_id'] = cluster_labels
    df_with_coords['pca_x'] = features_2d[:, 0]
    df_with_coords['pca_y'] = features_2d[:, 1]
    
    # Prepare papers data
    papers_data = []
    for _, paper in df_with_coords.iterrows():
        paper_dict = {
            'Key': paper.get('Key', ''),
            'Title': paper.get('Title', ''),
            'Author': paper.get('Author', ''),
            'summary': paper.get('summary', ''),
            'keywords': paper.get('keywords', ''),
            'cluster_id': int(paper['cluster_id']),
            'pca_x': float(paper['pca_x']),
            'pca_y': float(paper['pca_y']),
            'Publication Year': paper.get('Publication Year', ''),
            'DOI': paper.get('DOI', ''),
            'Url': paper.get('Url', '')
        }
        papers_data.append(paper_dict)
    
    # Prepare cluster info
    cluster_info = {}
    for cluster_id, analysis in cluster_analysis.items():
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
            'source_file': 'ASPLOS_labeled.csv',
            'clustering_method': 'K-means',
            'feature_extraction': 'TF-IDF'
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON data for web visualization saved to {output_path}")
    return json_data

def main():
    # Configuration
    csv_path = '/Users/cydia2001/Downloads/ASPLOS_labeled.csv'
    output_path = '/Users/cydia2001/Project/paper-labeler/clustered_papers.csv'
    json_output_path = '/Users/cydia2001/Project/paper-labeler/clustering_data.json'
    
    print("Loading ASPLOS papers dataset...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} papers")
    
    print("Preprocessing text data...")
    df = preprocess_text(df)
    print(f"After preprocessing: {len(df)} papers")
    
    print("Creating TF-IDF features...")
    features, vectorizer, feature_names = create_features(df)
    print(f"Created {features.shape[1]} features")
    
    print("Finding optimal number of clusters...")
    optimal_k, inertias, silhouette_scores, k_range = find_optimal_clusters(features)
    print(f"Optimal number of clusters: {optimal_k}")
    
    print("Performing clustering...")
    kmeans, cluster_labels = perform_clustering(features, optimal_k)
    
    print("Analyzing clusters...")
    cluster_analysis = analyze_clusters(df, cluster_labels, vectorizer, features)
    
    print("Creating visualizations...")
    fig1, pca, features_2d = visualize_clusters(features, cluster_labels, df, optimal_k)
    fig2 = plot_optimization_metrics(inertias, silhouette_scores, k_range, optimal_k)
    
    # Save results
    save_results(df, cluster_labels, cluster_analysis, output_path)
    
    # Generate JSON for web visualization
    print("Generating JSON data for web visualization...")
    json_data = generate_json_for_web(df, cluster_labels, cluster_analysis, features_2d, json_output_path)
    
    # Print summary
    print_cluster_summary(cluster_analysis)
    
    # Show plots
    plt.show()
    
    return df, cluster_labels, cluster_analysis, features, vectorizer

if __name__ == "__main__":
    df, cluster_labels, cluster_analysis, features, vectorizer = main()