"""
FastAPI endpoints for newspaper processing pipeline
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from pathlib import Path
import json

from .pdf_pipeline import process_pdf_to_jsonl
from .ner import extract_entities_from_jsonl
from .search import NewspaperSearcher, load_pages_from_jsonl

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Newspaper Processing API",
    description="API for processing multi-language newspaper PDFs",
    version="1.0.0"
)

# Global instances
searcher = NewspaperSearcher()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Newspaper Processing API",
        "version": "1.0.0",
        "endpoints": [
            "/convert",
            "/extract",
            "/search",
            "/context-search",
            "/report"
        ]
    }

@app.get("/convert")
async def convert_pdf(pdf_path: str = Query(..., description="Path to PDF file")):
    """
    Convert PDF to structured JSONL format
    """
    try:
        if not Path(pdf_path).exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        output_path = "data/output/pages.jsonl"
        pages_data = process_pdf_to_jsonl(pdf_path, output_path)
        
        # Create search index
        searcher.create_index(pages_data)
        
        return {
            "status": "success",
            "message": f"Converted {len(pages_data)} pages",
            "output_file": output_path,
            "pages_processed": len(pages_data)
        }
        
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/extract")
async def extract_entities(pages_path: str = Query(..., description="Path to pages JSONL file")):
    """
    Extract named entities from processed pages
    """
    try:
        if not Path(pages_path).exists():
            raise HTTPException(status_code=404, detail="Pages file not found")
        
        output_path = "data/output/entities.jsonl"
        entities = extract_entities_from_jsonl(pages_path, output_path)
        
        return {
            "status": "success",
            "message": f"Extracted {len(entities)} entities",
            "output_file": output_path,
            "entities_found": len(entities)
        }
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_keywords(q: str = Query(..., description="Search query")):
    """
    Perform keyword search across all pages
    """
    try:
        results = searcher.keyword_search(q)
        
        return {
            "status": "success",
            "query": q,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Keyword search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context-search")
async def context_search(keywords: str = Query(..., description="Comma-separated context keywords")):
    """
    Perform semantic search based on context keywords
    """
    try:
        keyword_list = [k.strip() for k in keywords.split(",")]
        
        # Load pages data for semantic search
        pages_path = "data/output/pages.jsonl"
        if not Path(pages_path).exists():
            raise HTTPException(status_code=404, detail="Pages file not found. Convert PDF first.")
        
        pages_data = load_pages_from_jsonl(pages_path)
        results = searcher.semantic_search(keyword_list, pages_data)
        
        return {
            "status": "success",
            "keywords": keyword_list,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def generate_report():
    """
    Generate processing report
    """
    try:
        pages_path = "data/output/pages.jsonl"
        entities_path = "data/output/entities.jsonl"
        
        pages_count = 0
        entities_count = 0
        
        if Path(pages_path).exists():
            with open(pages_path, 'r', encoding='utf-8') as f:
                pages_count = sum(1 for _ in f)
        
        if Path(entities_path).exists():
            with open(entities_path, 'r', encoding='utf-8') as f:
                entities_count = sum(1 for _ in f)
        
        return {
            "status": "success",
            "report": {
                "pages_processed": pages_count,
                "entities_extracted": entities_count,
                "index_created": searcher.index_dir.exists(),
                "api_status": "running"
            }
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Newspaper Processing API started")