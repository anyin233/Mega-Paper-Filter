#!/usr/bin/env python3
"""
LLM-based clustering analyzer for academic papers.
Uses OpenAI's language models to perform semantic clustering of papers based on titles and abstracts.
"""

import pandas as pd
import numpy as np
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from loguru import logger

from database import PaperDatabase

console = Console()

class LLMClusteringAnalyzer:
    """LLM-based clustering analyzer that uses semantic understanding for paper clustering."""
    
    def __init__(self, db_path: str, openai_config: Dict[str, Any]):
        """Initialize with database connection and OpenAI configuration."""
        self.db = PaperDatabase(db_path)
        self.openai_config = openai_config
        
        # Analysis results
        self.df = None
        self.clustering_result = None
        self.cluster_assignments = None
        
    def load_papers_from_db(self, dataset_name: str = None, min_papers: int = 5, max_papers: int = 100) -> pd.DataFrame:
        """Load papers from database for LLM-based analysis."""
        console.print(f"📚 Loading papers from database for LLM clustering...")
        
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
        
        # Filter out papers with insufficient content
        initial_count = len(self.df)
        
        # Require at least title
        self.df = self.df[self.df['title'].notna() & (self.df['title'].str.len() > 0)]
        
        # Filter to papers with abstracts for better LLM clustering
        self.df = self.df[self.df['abstract'].notna() & (self.df['abstract'].str.len() > 20)]
        
        filtered_count = len(self.df)
        
        if filtered_count < min_papers:
            raise ValueError(f"Insufficient papers for clustering: {filtered_count} (minimum: {min_papers})")
        
        if filtered_count < initial_count:
            console.print(f"⚠️  Filtered to {filtered_count} papers with abstracts (from {initial_count} total)")
        
        # Limit papers for LLM processing to avoid token limits and costs
        if filtered_count > max_papers:
            console.print(f"⚠️  Too many papers ({filtered_count}). Sampling {max_papers} papers for LLM clustering.")
            self.df = self.df.sample(n=max_papers, random_state=42).reset_index(drop=True)
            filtered_count = len(self.df)
        
        console.print(f"✅ Selected {filtered_count} papers for LLM clustering")
        
        # Display dataset information
        datasets = self.df['source_dataset'].value_counts()
        if len(datasets) > 1:
            console.print(f"📊 Papers by dataset:")
            for dataset, count in datasets.items():
                console.print(f"   • {dataset}: {count} papers")
        
        return self.df
    
    async def perform_llm_clustering(self, max_clusters: int = 8) -> Dict[str, Any]:
        """Perform LLM-based clustering of papers."""
        console.print("🤖 Performing LLM-based clustering...")
        
        try:
            # Import OpenAI functions
            import sys
            sys.path.append(str(Path(__file__).parent / "src"))
            from openai_api import get_async_openai_client, get_llm_clustering
            
            # Create async client
            client = get_async_openai_client(
                self.openai_config["api_key"], 
                self.openai_config.get("base_url", "https://api.openai.com/v1")
            )
            model = self.openai_config.get("model", "gpt-4o")
            
            # Prepare papers data for LLM
            papers_data = []
            for _, paper in self.df.iterrows():
                papers_data.append({
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'paper_id': paper.get('paper_id', paper.get('id', ''))
                })
            
            console.print(f"📤 Sending {len(papers_data)} papers to LLM for clustering...")
            console.print(f"🎯 Target clusters: {max_clusters}")
            console.print(f"🤖 Model: {model}")
            
            # Get LLM clustering result
            clustering_response = await get_llm_clustering(client, papers_data, max_clusters, model)
            
            # Use robust JSON parsing instead of direct json.loads
            import sys
            sys.path.append(str(Path(__file__).parent / "src"))
            from openai_api import safe_json_parse
            
            self.clustering_result = safe_json_parse(clustering_response, fallback_dict={
                "clusters": [],
                "unassigned_papers": list(range(len(papers_data))),
                "reasoning": "LLM clustering failed - JSON parsing error"
            })
            
            console.print(f"✅ LLM clustering completed!")
            console.print(f"📊 Created {len(self.clustering_result['clusters'])} clusters")
            
            # Process clustering results and create cluster assignments
            self.cluster_assignments = {}
            cluster_analysis = {}
            
            for cluster_idx, cluster in enumerate(self.clustering_result['clusters']):
                cluster_name = cluster['name']
                cluster_papers = cluster['papers']
                
                console.print(f"   🏷️  Cluster {cluster_idx}: {cluster_name} ({len(cluster_papers)} papers)")
                
                # Create cluster analysis data
                cluster_analysis[cluster_idx] = {
                    'name': cluster_name,
                    'size': len(cluster_papers),
                    'papers': cluster_papers,
                    'sample_titles': [p['title'] for p in cluster_papers[:5]]
                }
                
                # Assign papers to clusters based on title matching
                for cluster_paper in cluster_papers:
                    # Find matching paper in dataframe by title
                    matches = self.df[self.df['title'] == cluster_paper['title']]
                    if len(matches) > 0:
                        paper_id = matches.iloc[0]['paper_id'] if 'paper_id' in matches.columns else matches.iloc[0]['id']
                        self.cluster_assignments[paper_id] = cluster_idx
                    else:
                        # Fallback: try partial matching
                        partial_matches = self.df[self.df['title'].str.contains(cluster_paper['title'][:50], case=False, na=False)]
                        if len(partial_matches) > 0:
                            paper_id = partial_matches.iloc[0]['paper_id'] if 'paper_id' in partial_matches.columns else partial_matches.iloc[0]['id']
                            self.cluster_assignments[paper_id] = cluster_idx
                        else:
                            console.print(f"⚠️  Could not match paper: {cluster_paper['title'][:60]}...")
            
            # Add cluster assignments to dataframe
            self.df['cluster_id'] = self.df.apply(
                lambda row: self.cluster_assignments.get(row.get('paper_id', row.get('id')), -1), 
                axis=1
            )
            
            # Verify all papers were assigned
            unassigned = len(self.df[self.df['cluster_id'] == -1])
            if unassigned > 0:
                console.print(f"⚠️  {unassigned} papers could not be assigned to clusters")
            else:
                console.print("✅ All papers successfully assigned to clusters")
            
            return {
                'clustering_result': self.clustering_result,
                'cluster_analysis': cluster_analysis,
                'total_clusters': len(self.clustering_result['clusters']),
                'assigned_papers': len(self.df) - unassigned,
                'unassigned_papers': unassigned
            }
            
        except Exception as e:
            console.print(f"❌ Error during LLM clustering: {e}")
            raise
    
    def generate_visualization_data(self, output_path: str, dataset_name: str = None) -> Dict[str, Any]:
        """Generate visualization data for web interface."""
        console.print("📄 Generating visualization data...")
        
        if self.clustering_result is None:
            raise ValueError("No clustering result available. Run perform_llm_clustering() first.")
        
        # Create 2D coordinates for visualization (simple grid layout since we don't have PCA)
        papers_data = []
        cluster_positions = {}
        
        # Calculate positions for each cluster
        num_clusters = len(self.clustering_result['clusters'])
        cols = int(np.ceil(np.sqrt(num_clusters)))
        rows = int(np.ceil(num_clusters / cols))
        
        for i, cluster in enumerate(self.clustering_result['clusters']):
            cluster_x = (i % cols) * 2.0 - cols + 1
            cluster_y = (i // cols) * 2.0 - rows + 1
            cluster_positions[i] = (cluster_x, cluster_y)
        
        # Prepare papers data
        for _, paper in self.df.iterrows():
            cluster_id = paper.get('cluster_id', -1)
            
            if cluster_id >= 0 and cluster_id in cluster_positions:
                base_x, base_y = cluster_positions[cluster_id]
                # Add small random offset within cluster
                offset_x = np.random.uniform(-0.3, 0.3)
                offset_y = np.random.uniform(-0.3, 0.3)
                pca_x = base_x + offset_x
                pca_y = base_y + offset_y
            else:
                # Unassigned papers go to origin
                pca_x, pca_y = 0.0, 0.0
            
            paper_dict = {
                'Key': str(paper.get('paper_id', paper.get('id', ''))),
                'Title': paper.get('title', ''),
                'Author': paper.get('authors_str', ''),
                'summary': paper.get('summary', ''),
                'keywords': ', '.join(paper.get('keywords_parsed', [])),
                'cluster_id': int(cluster_id) if cluster_id >= 0 else 0,
                'pca_x': float(pca_x),
                'pca_y': float(pca_y),
                'Publication Year': paper.get('publication_year', ''),
                'DOI': paper.get('doi', ''),
                'Url': paper.get('url', ''),
                'Venue': paper.get('venue', ''),
                'Abstract': paper.get('abstract', '')
            }
            papers_data.append(paper_dict)
        
        # Prepare cluster info
        cluster_info = {}
        for cluster_idx, cluster in enumerate(self.clustering_result['clusters']):
            cluster_papers_in_df = self.df[self.df['cluster_id'] == cluster_idx]
            
            # Get keywords from papers in this cluster
            all_keywords = []
            for _, paper in cluster_papers_in_df.iterrows():
                all_keywords.extend(paper.get('keywords_parsed', []))
            
            # Create keyword frequency
            keyword_freq = pd.Series(all_keywords).value_counts()
            common_keywords = keyword_freq.head(10).to_dict()
            
            # Use LLM-generated cluster titles as top keywords with high scores
            cluster_name_words = cluster['name'].split()
            top_keywords = [(word, 1.0) for word in cluster_name_words]
            
            # Add common paper keywords with lower scores
            for keyword, count in list(common_keywords.items())[:7]:
                if keyword not in cluster_name_words:
                    top_keywords.append((keyword, count / len(cluster_papers_in_df)))
            
            cluster_info[str(cluster_idx)] = {
                'name': cluster['name'],
                'description': f"LLM-identified cluster: {cluster['name']}",
                'size': len(cluster_papers_in_df),
                'top_keywords': top_keywords[:10],
                'sample_titles': [p['title'] for p in cluster['papers'][:5]],
                'common_paper_keywords': common_keywords
            }
        
        # Create metadata
        db_stats = self.db.get_statistics()
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'source_database': str(self.db.db_path),
            'dataset_filter': dataset_name,
            'clustering_method': 'LLM-based Semantic Clustering',
            'feature_extraction': 'OpenAI Language Model',
            'total_papers_in_db': db_stats['total_papers'],
            'papers_used_for_clustering': len(papers_data),
            'pca_explained_variance': [1.0, 1.0],  # Not applicable for LLM clustering
            'llm_model': self.openai_config.get('model', 'gpt-4o'),
            'max_clusters_requested': len(cluster_info)
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
        if self.clustering_result is None:
            console.print("❌ No clustering result available")
            return
            
        console.print(Panel.fit(
            "[bold blue]LLM CLUSTERING SUMMARY[/bold blue]",
            border_style="blue"
        ))
        
        for cluster_idx, cluster in enumerate(self.clustering_result['clusters']):
            cluster_name = cluster['name']
            cluster_papers = cluster['papers']
            
            console.print(f"\n[bold cyan]Cluster {cluster_idx}: {cluster_name}[/bold cyan] ({len(cluster_papers)} papers)")
            console.print("─" * 60)
            
            # Show sample paper titles
            console.print("[yellow]Papers in this cluster:[/yellow]")
            for i, paper in enumerate(cluster_papers[:5], 1):
                console.print(f"  {i}. {paper['title']}")
            
            if len(cluster_papers) > 5:
                console.print(f"  ... and {len(cluster_papers) - 5} more papers")
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

def main():
    """Command-line interface for LLM-based clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster papers using LLM semantic analysis")
    parser.add_argument('--database', default='papers.db', help='Database file path')
    parser.add_argument('--dataset', help='Dataset name to filter (optional)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--output-prefix', default='llm_clustering', help='Output file prefix')
    parser.add_argument('--max-clusters', type=int, default=8, help='Maximum number of clusters')
    parser.add_argument('--max-papers', type=int, default=100, help='Maximum papers to process')
    parser.add_argument('--openai-api-key', required=True, help='OpenAI API key')
    parser.add_argument('--openai-base-url', default='https://api.openai.com/v1', help='OpenAI base URL')
    parser.add_argument('--openai-model', default='gpt-4o', help='OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, or any custom model name)')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.database).exists():
        console.print(f"❌ Database not found: {args.database}")
        console.print("💡 Run parse_paper_db.py first to populate the database")
        return
    
    # Set up OpenAI configuration
    openai_config = {
        'api_key': args.openai_api_key,
        'base_url': args.openai_base_url,
        'model': args.openai_model
    }
    
    # Initialize analyzer
    analyzer = LLMClusteringAnalyzer(args.database, openai_config)
    
    try:
        # Display banner
        console.print(Panel.fit(
            f"[bold blue]LLM-Based Clustering Analysis[/bold blue]\n"
            f"🗃️  Database: {args.database}\n" +
            (f"📊 Dataset filter: {args.dataset}\n" if args.dataset else "") +
            f"📁 Output directory: {args.output_dir}\n"
            f"🤖 Model: {args.openai_model}\n"
            f"🎯 Max clusters: {args.max_clusters}",
            border_style="blue"
        ))
        
        # Run analysis
        async def run_analysis():
            # Load papers
            df = analyzer.load_papers_from_db(args.dataset, max_papers=args.max_papers)
            
            # Perform LLM clustering
            result = await analyzer.perform_llm_clustering(args.max_clusters)
            
            # Generate outputs
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            output_prefix = output_dir / args.output_prefix
            
            # Generate visualization data
            json_output = f"{output_prefix}_data.json"
            analyzer.generate_visualization_data(json_output, args.dataset)
            
            # Print summary
            analyzer.print_cluster_summary()
            
            # Show completion message
            console.print(f"\n[bold green]✅ LLM Clustering Complete![/bold green]")
            console.print(f"📄 Visualization data: {json_output}")
            console.print(f"🤖 Generated {result['total_clusters']} clusters using {args.openai_model}")
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