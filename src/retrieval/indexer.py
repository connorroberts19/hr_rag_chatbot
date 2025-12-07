"""Indexing pipeline to build the vector store from documents."""

from src.ingestion.pipeline import load_chunks, run_ingestion_pipeline
from src.retrieval.vector_store import add_documents, clear_vector_store, get_vector_store


def build_index(
    force_reprocess: bool = False,
    clear_existing: bool = True,
) -> None:
    """
    Build the vector store index from HR documents.

    Args:
        force_reprocess: If True, re-run document ingestion even if chunks exist
        clear_existing: If True, clear existing vector store before indexing
    """
    print("=" * 50)
    print("Building Vector Store Index")
    print("=" * 50)

    # Step 1: Get or create chunks
    if force_reprocess:
        print("\n[1/2] Processing documents...")
        chunks = run_ingestion_pipeline()
    else:
        print("\n[1/2] Loading processed chunks...")
        chunks = load_chunks()

        if not chunks:
            print("No processed chunks found. Running ingestion pipeline...")
            chunks = run_ingestion_pipeline()

    if not chunks:
        print("No documents to index!")
        return

    # Step 2: Build vector store
    print(f"\n[2/2] Indexing {len(chunks)} chunks...")

    if clear_existing:
        clear_vector_store()

    add_documents(chunks)

    print("\n" + "=" * 50)
    print("Index build complete!")
    print("=" * 50)


def get_index_stats() -> dict:
    """Get statistics about the current index."""
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        count = collection.count()

        return {
            "document_count": count,
            "collection_name": "hr_documents",
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    build_index(force_reprocess=force)

    print("\n--- Index Stats ---")
    stats = get_index_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
