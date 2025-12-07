"""Main ingestion pipeline that orchestrates document processing."""

import json
from pathlib import Path

from langchain_core.documents import Document

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.ingestion.document_loader import load_all_documents
from src.ingestion.text_processor import (
    chunk_markdown_by_headers,
    enrich_metadata,
)


def run_ingestion_pipeline(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    save_to_disk: bool = True,
) -> list[Document]:
    """
    Run the full document ingestion pipeline.

    Steps:
    1. Load all documents from input directory
    2. Enrich metadata (categorization, etc.)
    3. Chunk documents for embedding
    4. Optionally save processed chunks to disk

    Returns:
        List of processed document chunks ready for embedding
    """
    print("=" * 50)
    print("Starting Document Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load documents
    print("\n[1/3] Loading documents...")
    documents = load_all_documents(input_dir)

    if not documents:
        print("No documents found! Add documents to data/raw/")
        return []

    # Step 2: Enrich metadata
    print("\n[2/3] Enriching metadata...")
    documents = enrich_metadata(documents)

    # Step 3: Chunk documents
    print("\n[3/3] Chunking documents...")
    chunks = chunk_markdown_by_headers(documents)

    # Save to disk if requested
    if save_to_disk:
        save_chunks(chunks, output_dir)

    print("\n" + "=" * 50)
    print(f"Pipeline complete! Processed {len(chunks)} chunks")
    print("=" * 50)

    return chunks


def save_chunks(chunks: list[Document], output_dir: Path) -> None:
    """Save processed chunks to JSON for inspection/debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"

    # Convert to serializable format
    chunks_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(output_file, "w") as f:
        json.dump(chunks_data, f, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks(input_dir: Path = PROCESSED_DATA_DIR) -> list[Document]:
    """Load previously processed chunks from disk."""
    input_file = input_dir / "chunks.json"

    if not input_file.exists():
        print(f"No processed chunks found at {input_file}")
        return []

    with open(input_file) as f:
        chunks_data = json.load(f)

    chunks = [
        Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks_data
    ]

    print(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks
