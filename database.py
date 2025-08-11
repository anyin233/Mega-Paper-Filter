#!/usr/bin/env python3
"""
Database module for storing and managing academic papers data.
Provides SQLite-based storage for the paper analysis pipeline.
"""
import sqlite3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib

class PaperDatabase:
    """SQLite database for storing academic papers with essential metadata."""
    
    def __init__(self, db_path: str = "papers.db"):
        """
        Initialize the database connection and create tables if needed.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
    def create_tables(self):
        """Create the papers and datasets tables if they don't exist."""
        with self.conn:
            # Main papers table with essential fields
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT UNIQUE,  -- Original paper ID from source
                    title TEXT NOT NULL,
                    authors TEXT,  -- JSON array of authors
                    abstract TEXT,
                    url TEXT,
                    doi TEXT,
                    publication_year INTEGER,
                    venue TEXT,  -- Journal/conference name
                    summary TEXT,  -- AI-generated summary
                    keywords TEXT,  -- JSON array of keywords
                    content_hash TEXT,  -- Hash of title+abstract for deduplication
                    source_dataset TEXT,  -- Which dataset this came from
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Datasets metadata table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    source_file TEXT,
                    total_papers INTEGER DEFAULT 0,
                    processed_papers INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_paper_id ON papers(paper_id)
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_content_hash ON papers(content_hash)
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_source_dataset ON papers(source_dataset)
            ''')
    
    def _generate_content_hash(self, title: str, abstract: str = "") -> str:
        """Generate hash for deduplication based on title and abstract."""
        content = f"{title.lower().strip()}{abstract.lower().strip()}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def create_dataset(self, name: str, description: str = "", source_file: str = "") -> int:
        """
        Create a new dataset record.
        
        Args:
            name: Dataset name
            description: Dataset description
            source_file: Source CSV file path
            
        Returns:
            Dataset ID
        """
        with self.conn:
            cursor = self.conn.execute('''
                INSERT OR REPLACE INTO datasets (name, description, source_file, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (name, description, source_file))
            return cursor.lastrowid
    
    def get_or_create_dataset(self, name: str, description: str = "", source_file: str = "") -> int:
        """Get existing dataset or create new one."""
        cursor = self.conn.execute('SELECT id FROM datasets WHERE name = ?', (name,))
        row = cursor.fetchone()
        
        if row:
            return row['id']
        else:
            return self.create_dataset(name, description, source_file)
    
    def add_paper(self, 
                  paper_id: str,
                  title: str,
                  authors: List[str] = None,
                  abstract: str = "",
                  url: str = "",
                  doi: str = "",
                  publication_year: int = None,
                  venue: str = "",
                  summary: str = "",
                  keywords: List[str] = None,
                  source_dataset: str = "default") -> Optional[int]:
        """
        Add a paper to the database.
        
        Args:
            paper_id: Unique paper identifier
            title: Paper title
            authors: List of author names
            abstract: Paper abstract
            url: Paper URL
            doi: DOI identifier
            publication_year: Publication year
            venue: Journal/conference name
            summary: AI-generated summary
            keywords: List of keywords
            source_dataset: Source dataset name
            
        Returns:
            Paper database ID if successful, None if duplicate
        """
        if not title.strip():
            return None
            
        authors_json = json.dumps(authors or [])
        keywords_json = json.dumps(keywords or [])
        content_hash = self._generate_content_hash(title, abstract)
        
        try:
            with self.conn:
                cursor = self.conn.execute('''
                    INSERT INTO papers (
                        paper_id, title, authors, abstract, url, doi,
                        publication_year, venue, summary, keywords,
                        content_hash, source_dataset, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (paper_id, title, authors_json, abstract, url, doi,
                      publication_year, venue, summary, keywords_json,
                      content_hash, source_dataset))
                
                return cursor.lastrowid
                
        except sqlite3.IntegrityError:
            # Paper already exists (duplicate paper_id or content_hash)
            return None
    
    def update_paper_summary(self, paper_id: str, summary: str, keywords: List[str] = None):
        """Update the AI-generated summary and keywords for a paper."""
        keywords_json = json.dumps(keywords or [])
        
        with self.conn:
            self.conn.execute('''
                UPDATE papers 
                SET summary = ?, keywords = ?, updated_at = CURRENT_TIMESTAMP
                WHERE paper_id = ?
            ''', (summary, keywords_json, paper_id))
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get a single paper by ID."""
        cursor = self.conn.execute('''
            SELECT * FROM papers WHERE paper_id = ?
        ''', (paper_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_papers_by_dataset(self, dataset_name: str) -> List[Dict]:
        """Get all papers from a specific dataset."""
        cursor = self.conn.execute('''
            SELECT * FROM papers 
            WHERE source_dataset = ?
            ORDER BY created_at DESC
        ''', (dataset_name,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_papers(self) -> List[Dict]:
        """Get all papers in the database."""
        cursor = self.conn.execute('''
            SELECT * FROM papers ORDER BY created_at DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def search_papers(self, query: str, dataset_name: str = None) -> List[Dict]:
        """
        Search papers by title, abstract, or keywords.
        
        Args:
            query: Search query
            dataset_name: Optional dataset filter
            
        Returns:
            List of matching papers
        """
        search_term = f"%{query}%"
        
        if dataset_name:
            cursor = self.conn.execute('''
                SELECT * FROM papers 
                WHERE source_dataset = ? AND (
                    title LIKE ? OR abstract LIKE ? OR summary LIKE ? OR keywords LIKE ?
                )
                ORDER BY created_at DESC
            ''', (dataset_name, search_term, search_term, search_term, search_term))
        else:
            cursor = self.conn.execute('''
                SELECT * FROM papers 
                WHERE title LIKE ? OR abstract LIKE ? OR summary LIKE ? OR keywords LIKE ?
                ORDER BY created_at DESC
            ''', (search_term, search_term, search_term, search_term))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_datasets(self) -> List[Dict]:
        """Get all datasets with statistics."""
        cursor = self.conn.execute('''
            SELECT d.*, COUNT(p.id) as actual_papers
            FROM datasets d
            LEFT JOIN papers p ON d.name = p.source_dataset
            GROUP BY d.id
            ORDER BY d.created_at DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def update_dataset_stats(self, dataset_name: str):
        """Update paper counts for a dataset."""
        cursor = self.conn.execute('''
            SELECT COUNT(*) as count FROM papers WHERE source_dataset = ?
        ''', (dataset_name,))
        
        count = cursor.fetchone()['count']
        
        with self.conn:
            self.conn.execute('''
                UPDATE datasets 
                SET processed_papers = ?, updated_at = CURRENT_TIMESTAMP
                WHERE name = ?
            ''', (count, dataset_name))
    
    def export_to_dataframe(self, dataset_name: str = None) -> pd.DataFrame:
        """
        Export papers to pandas DataFrame for analysis.
        
        Args:
            dataset_name: Optional dataset filter
            
        Returns:
            DataFrame with paper data
        """
        if dataset_name:
            papers = self.get_papers_by_dataset(dataset_name)
        else:
            papers = self.get_all_papers()
        
        if not papers:
            return pd.DataFrame()
        
        # Convert to DataFrame and parse JSON fields
        df = pd.DataFrame(papers)
        
        # Parse JSON fields
        df['authors_parsed'] = df['authors'].apply(
            lambda x: json.loads(x) if x else []
        )
        df['keywords_parsed'] = df['keywords'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        # Create combined text field for analysis
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['abstract'].fillna('') + ' ' + 
            df['summary'].fillna('') + ' ' + 
            df['keywords'].fillna('')
        )
        
        return df
    
    def export_to_csv(self, output_path: str, dataset_name: str = None):
        """Export papers to CSV file."""
        df = self.export_to_dataframe(dataset_name)
        
        if df.empty:
            print("No papers to export")
            return
        
        # Flatten JSON fields for CSV export
        df_export = df.copy()
        df_export['authors'] = df_export['authors_parsed'].apply(
            lambda x: '; '.join(x) if x else ''
        )
        df_export['keywords'] = df_export['keywords_parsed'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        
        # Select and rename columns for export
        columns_to_export = [
            'paper_id', 'title', 'authors', 'abstract', 'url', 'doi',
            'publication_year', 'venue', 'summary', 'keywords', 'source_dataset'
        ]
        
        df_export = df_export[columns_to_export]
        df_export.to_csv(output_path, index=False)
        print(f"Exported {len(df_export)} papers to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.execute('''
            SELECT 
                COUNT(*) as total_papers,
                COUNT(DISTINCT source_dataset) as total_datasets,
                COUNT(CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 END) as papers_with_summary,
                COUNT(CASE WHEN keywords IS NOT NULL AND keywords != '[]' THEN 1 END) as papers_with_keywords
            FROM papers
        ''')
        
        stats = dict(cursor.fetchone())
        
        # Get dataset breakdown
        cursor = self.conn.execute('''
            SELECT source_dataset, COUNT(*) as count
            FROM papers
            GROUP BY source_dataset
            ORDER BY count DESC
        ''')
        
        stats['datasets_breakdown'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def cleanup_duplicates(self) -> int:
        """Remove duplicate papers based on content hash."""
        with self.conn:
            cursor = self.conn.execute('''
                DELETE FROM papers 
                WHERE id NOT IN (
                    SELECT MIN(id) 
                    FROM papers 
                    GROUP BY content_hash
                )
            ''')
            return cursor.rowcount
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Example usage and testing."""
    # Create database and add sample data
    db = PaperDatabase("test_papers.db")
    
    # Create a dataset
    dataset_id = db.create_dataset(
        "Test Dataset", 
        "Sample papers for testing",
        "test_papers.csv"
    )
    
    # Add sample papers
    papers = [
        {
            'paper_id': 'test001',
            'title': 'Machine Learning in Computer Vision',
            'authors': ['John Doe', 'Jane Smith'],
            'abstract': 'This paper explores the use of machine learning techniques in computer vision applications.',
            'url': 'https://example.com/paper1',
            'doi': '10.1000/test001',
            'publication_year': 2023,
            'venue': 'IEEE Computer Vision Conference',
            'summary': 'An exploration of ML techniques for CV applications with promising results.',
            'keywords': ['machine learning', 'computer vision', 'deep learning'],
            'source_dataset': 'Test Dataset'
        },
        {
            'paper_id': 'test002',
            'title': 'Natural Language Processing Advances',
            'authors': ['Alice Johnson', 'Bob Wilson'],
            'abstract': 'Recent advances in natural language processing using transformer models.',
            'url': 'https://example.com/paper2',
            'publication_year': 2024,
            'venue': 'ACL Conference',
            'summary': 'Discussion of transformer model improvements in NLP tasks.',
            'keywords': ['NLP', 'transformers', 'language models'],
            'source_dataset': 'Test Dataset'
        }
    ]
    
    # Add papers to database
    for paper in papers:
        paper_id = db.add_paper(**paper)
        print(f"Added paper with ID: {paper_id}")
    
    # Update dataset statistics
    db.update_dataset_stats('Test Dataset')
    
    # Display statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total datasets: {stats['total_datasets']}")
    print(f"Papers with summaries: {stats['papers_with_summary']}")
    print(f"Papers with keywords: {stats['papers_with_keywords']}")
    
    # Search papers
    results = db.search_papers('machine learning')
    print(f"\nFound {len(results)} papers matching 'machine learning'")
    
    # Export to DataFrame
    df = db.export_to_dataframe('Test Dataset')
    print(f"\nExported DataFrame with {len(df)} rows")
    
    # Export to CSV
    db.export_to_csv('test_export.csv', 'Test Dataset')
    
    db.close()


if __name__ == "__main__":
    main()