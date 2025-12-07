"""Vector store management using ChromaDB."""

from pathlib import Path
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DB_DIR, TOP_K_RESULTS
from src.retrieval.embeddings import get_embedding_model


def get_vector_store(
    collection_name: str = "hr_documents",
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """
    Get or create a ChromaDB vector store.

    Args:
        collection_name: Name of the collection in ChromaDB
        persist_directory: Directory to persist the database

    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR

    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def add_documents(
    documents: list[Document],
    collection_name: str = "hr_documents",
) -> Chroma:
    """
    Add documents to the vector store.

    Args:
        documents: List of Document objects to add
        collection_name: Name of the collection

    Returns:
        The vector store with added documents
    """
    vector_store = get_vector_store(collection_name)

    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"Added batch {i // batch_size + 1}: {len(batch)} documents")

    print(f"\nTotal documents in vector store: {len(documents)}")
    return vector_store


def similarity_search(
    query: str,
    k: int = TOP_K_RESULTS,
    collection_name: str = "hr_documents",
    filter_dict: Optional[dict] = None,
) -> list[Document]:
    """
    Search for similar documents.

    Args:
        query: Search query string
        k: Number of results to return
        collection_name: Name of the collection to search
        filter_dict: Optional metadata filter

    Returns:
        List of similar documents
    """
    vector_store = get_vector_store(collection_name)

    if filter_dict:
        results = vector_store.similarity_search(query, k=k, filter=filter_dict)
    else:
        results = vector_store.similarity_search(query, k=k)

    return results


def similarity_search_with_scores(
    query: str,
    k: int = TOP_K_RESULTS,
    collection_name: str = "hr_documents",
) -> list[tuple[Document, float]]:
    """
    Search for similar documents with relevance scores.

    Args:
        query: Search query string
        k: Number of results to return
        collection_name: Name of the collection to search

    Returns:
        List of (document, score) tuples
    """
    vector_store = get_vector_store(collection_name)
    return vector_store.similarity_search_with_score(query, k=k)


def get_retriever(
    k: int = TOP_K_RESULTS,
    collection_name: str = "hr_documents",
):
    """
    Get a retriever for use in RAG chains.

    Args:
        k: Number of documents to retrieve
        collection_name: Name of the collection

    Returns:
        A retriever instance
    """
    vector_store = get_vector_store(collection_name)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def clear_vector_store(collection_name: str = "hr_documents") -> None:
    """Delete all documents from the vector store."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(collection_name)
        print(f"Cleared collection: {collection_name}")
    except Exception:
        print(f"Collection {collection_name} does not exist or already cleared")
