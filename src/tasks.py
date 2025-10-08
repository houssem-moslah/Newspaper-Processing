"""
CLI entrypoint for newspaper processing pipeline
Uses Typer for clean command-line interface
"""

import typer
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Optional, List
import uvicorn
import json

from .pdf_pipeline import process_pdf_to_jsonl
from .ner import extract_entities_from_jsonl
from .search import NewspaperSearcher, load_pages_from_jsonl
from .api import app as fastapi_app

# Initialize Typer app
app = typer.Typer(
    name="newspaper-pipeline",
    help="Multi-language Newspaper Processing Pipeline"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_task_start(task_name: str):
    """Log task start with memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[{task_name}] Starting task (Memory: {memory_mb:.1f} MB)")

def log_task_end(task_name: str, start_time: float):
    """Log task completion with duration and memory usage"""
    duration = time.time() - start_time
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[{task_name}] Completed in {duration:.2f}s (Memory: {memory_mb:.1f} MB)")

@app.command()
def convert(
    pdf_path: str = typer.Argument(..., help="Path to input PDF file"),
    output_path: str = typer.Option("data/output/pages.jsonl", help="Output JSONL file path")
):
    """
    Convert PDF to structured pages JSONL
    """
    start_time = time.time()
    log_task_start("CONVERT")
    
    try:
        pages_data = process_pdf_to_jsonl(pdf_path, output_path)
        typer.echo(f"✅ Converted {len(pages_data)} pages to {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        typer.echo(f"❌ Conversion failed: {str(e)}")
        raise typer.Exit(1)
    
    log_task_end("CONVERT", start_time)

@app.command()
def extract(
    input_path: str = typer.Argument(..., help="Path to pages JSONL file"),
    output_path: str = typer.Option("data/output/entities.jsonl", help="Output entities file path")
):
    """
    Extract named entities from pages
    """
    start_time = time.time()
    log_task_start("EXTRACT")
    
    try:
        entities = extract_entities_from_jsonl(input_path, output_path)
        typer.echo(f"✅ Extracted {len(entities)} entities to {output_path}")
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        typer.echo(f"❌ Entity extraction failed: {str(e)}")
        raise typer.Exit(1)
    
    log_task_end("EXTRACT", start_time)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, help="Maximum number of results")
):
    """
    Search for keywords across all pages
    """
    start_time = time.time()
    log_task_start("SEARCH")
    
    try:
        searcher = NewspaperSearcher()
        results = searcher.keyword_search(query, limit=limit)
        
        if results:
            typer.echo(f"🔍 Found {len(results)} matches for '{query}':")
            for i, result in enumerate(results, 1):
                typer.echo(f"{i}. Page {result['page_num']}: {result['snippet']}")
        else:
            typer.echo(f"❌ No matches found for '{query}'")
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        typer.echo(f"❌ Search failed: {str(e)}")
        raise typer.Exit(1)
    
    log_task_end("SEARCH", start_time)

@app.command()
def context(
    keywords: str = typer.Argument(..., help="Comma-separated context keywords"),
    top_k: int = typer.Option(5, help="Number of top results")
):
    """
    Find paragraphs semantically similar to context keywords
    """
    start_time = time.time()
    log_task_start("CONTEXT")
    
    try:
        keyword_list = [k.strip() for k in keywords.split(",")]
        searcher = NewspaperSearcher()
        
        # Load pages data
        pages_path = "data/output/pages.jsonl"
        if not Path(pages_path).exists():
            typer.echo("❌ Pages file not found. Run convert command first.")
            raise typer.Exit(1)
            
        pages_data = load_pages_from_jsonl(pages_path)
        results = searcher.semantic_search(keyword_list, pages_data, top_k=top_k)
        
        if results:
            typer.echo(f"🎯 Top {len(results)} related paragraphs:")
            for i, result in enumerate(results, 1):
                typer.echo(f"{i}. \"{result['paragraph'][:100]}...\"")
                typer.echo(f"   (Page {result['page_num']}, Similarity: {result['similarity']})")
                typer.echo()
        else:
            typer.echo("❌ No related paragraphs found")
            
    except Exception as e:
        logger.error(f"Context search failed: {str(e)}")
        typer.echo(f"❌ Context search failed: {str(e)}")
        raise typer.Exit(1)
    
    log_task_end("CONTEXT", start_time)

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(7070, help="Port to bind")
):
    """
    Start FastAPI server
    """
    log_task_start("SERVE")
    typer.echo(f"🚀 Starting API server on {host}:{port}")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

@app.command()
def run_all(
    pdf_path: str = typer.Argument(..., help="Path to input PDF file")
):
    """
    Run complete pipeline: convert → extract → search index
    """
    start_time = time.time()
    log_task_start("FULL_PIPELINE")
    
    try:
        # Step 1: Convert PDF
        typer.echo("📄 Step 1: Converting PDF to structured pages...")
        pages_data = process_pdf_to_jsonl(pdf_path, "data/output/pages.jsonl")
        
        # Step 2: Extract entities
        typer.echo("🏷️  Step 2: Extracting named entities...")
        entities = extract_entities_from_jsonl("data/output/pages.jsonl", "data/output/entities.jsonl")
        
        # Step 3: Create search index
        typer.echo("🔍 Step 3: Creating search index...")
        searcher = NewspaperSearcher()
        searcher.create_index(pages_data)
        
        typer.echo(f"✅ Pipeline completed successfully!")
        typer.echo(f"   - Pages processed: {len(pages_data)}")
        typer.echo(f"   - Entities extracted: {len(entities)}")
        typer.echo(f"   - Search index created")
        typer.echo(f"   - Ready for search and API operations")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        typer.echo(f"❌ Pipeline failed: {str(e)}")
        raise typer.Exit(1)
    
    log_task_end("FULL_PIPELINE", start_time)

if __name__ == "__main__":
    app()