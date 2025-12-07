"""Document ingestion module for loading and processing HR documents."""

from src.ingestion.document_loader import load_all_documents
from src.ingestion.pipeline import load_chunks, run_ingestion_pipeline
from src.ingestion.text_processor import chunk_markdown_by_headers

__all__ = [
    "load_all_documents",
    "run_ingestion_pipeline",
    "load_chunks",
    "chunk_markdown_by_headers",
]
