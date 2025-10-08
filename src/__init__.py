"""
Newspaper Processing Pipeline
A modular system for processing multi-language newspaper PDFs
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from . import pdf_pipeline, ner, search, api, tasks

__all__ = ["pdf_pipeline", "ner", "search", "api", "tasks"]