"""Retrieval module for vector store and semantic search."""

from src.retrieval.embeddings import get_embedding_model
from src.retrieval.indexer import build_index, get_index_stats
from src.retrieval.vector_store import (
    add_documents,
    get_retriever,
    get_vector_store,
    similarity_search,
    similarity_search_with_scores,
)

__all__ = [
    "get_embedding_model",
    "build_index",
    "get_index_stats",
    "add_documents",
    "get_retriever",
    "get_vector_store",
    "similarity_search",
    "similarity_search_with_scores",
]
