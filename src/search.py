"""
Search functionality for newspaper content
Includes both keyword search (Whoosh) and semantic search
"""

import logging
from typing import List, Dict, Any, Tuple
from whoosh import index
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configure logging
logger = logging.getLogger(__name__)

class NewspaperSearcher:
    """
    Handles both keyword and semantic search over newspaper content
    """
    
    def __init__(self, index_dir: str = "data/search_index"):
        """
        Initialize search functionality
        
        Args:
            index_dir: Directory for search index
        """
        self.index_dir = Path(index_dir)
        self.schema = Schema(
            page_num=NUMERIC(stored=True),
            content_fr=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            content_ar=TEXT(stored=True),
            content_markdown=TEXT(stored=True)
        )
        
        # Initialize semantic model lazily
        self.semantic_model = None
        
    def create_index(self, pages_data: List[Dict]):
        """
        Create search index from pages data
        
        Args:
            pages_data: List of page dictionaries
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index
        ix = index.create_in(str(self.index_dir), self.schema)
        writer = ix.writer()
        
        for page in pages_data:
            writer.add_document(
                page_num=page["page_num"],
                content_fr=page["page_content_fr"],
                content_ar=page["page_content_arabic"],
                content_markdown=page["page_content_markdown"]
            )
        
        writer.commit()
        logger.info(f"Created search index with {len(pages_data)} documents")
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform keyword search using Whoosh
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of search results with context
        """
        try:
            if not self.index_dir.exists() or not index.exists_in(str(self.index_dir)):
                raise ValueError("Search index not found. Please create index first.")
            
            ix = index.open_dir(str(self.index_dir))
            results = []
            
            with ix.searcher() as searcher:
                query_parser = QueryParser("content_fr", ix.schema)
                parsed_query = query_parser.parse(query)
                
                search_results = searcher.search(parsed_query, limit=limit)
                
                for hit in search_results:
                    # Extract context snippet
                    content = hit["content_fr"]
                    words = content.split()
                    query_words = query.lower().split()
                    
                    # Find position of first query word
                    snippet_start = 0
                    for i, word in enumerate(words):
                        if any(qw in word.lower() for qw in query_words):
                            snippet_start = max(0, i - 10)  # 10 words before
                            break
                    
                    snippet = " ".join(words[snippet_start:snippet_start + 30])
                    if snippet_start > 0:
                        snippet = "..." + snippet
                    if len(words) > snippet_start + 30:
                        snippet = snippet + "..."
                    
                    results.append({
                        "page_num": hit["page_num"],
                        "snippet": snippet,
                        "score": hit.score,
                        "full_content": content[:200] + "..." if len(content) > 200 else content
                    })
            
            logger.info(f"Keyword search '{query}' found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def semantic_search(self, keywords: List[str], pages_data: List[Dict], 
                       top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using sentence embeddings
        
        Args:
            keywords: List of context keywords
            pages_data: Pages data for search
            top_k: Number of top results to return
            
        Returns:
            List of semantic search results
        """
        try:
            # Initialize model if not already loaded
            if self.semantic_model is None:
                logger.info("Loading semantic model...")
                self.semantic_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
            
            # Create query embedding
            query_text = " ".join(keywords)
            query_embedding = self.semantic_model.encode([query_text])
            
            # Extract paragraphs and create embeddings
            paragraphs = []
            paragraph_info = []  # Store (page_num, paragraph_text)
            
            for page in pages_data:
                # Split French content into paragraphs
                page_paragraphs = [p.strip() for p in page["page_content_fr"].split('\n') 
                                 if p.strip() and len(p.strip()) > 20]
                
                for para in page_paragraphs:
                    paragraphs.append(para)
                    paragraph_info.append((page["page_num"], para))
            
            if not paragraphs:
                logger.warning("No paragraphs found for semantic search")
                return []
            
            # Encode paragraphs
            para_embeddings = self.semantic_model.encode(paragraphs)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, para_embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                page_num, paragraph = paragraph_info[idx]
                similarity = float(similarities[idx])
                
                results.append({
                    "page_num": page_num,
                    "paragraph": paragraph,
                    "similarity": round(similarity, 3)
                })
            
            logger.info(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

# Utility functions
def load_pages_from_jsonl(file_path: str) -> List[Dict]:
    """Load pages data from JSONL file"""
    pages_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pages_data.append(json.loads(line))
    return pages_data